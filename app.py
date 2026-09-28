import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from pathlib import Path

try:
    from google import genai
except ImportError:
    genai = None

st.set_page_config(page_title='CleanData AI', page_icon='🧹', layout='wide', initial_sidebar_state='expanded')

st.markdown('''<style>
.main-title{font-size:42px;font-weight:700;color:#1557C0}.subtitle{font-size:22px;font-weight:600;color:#263238}.section-title{font-size:28px;font-weight:700;color:#1557C0}.small-text{color:#5F6B7A;font-size:16px;line-height:1.6}.feature-card{padding:22px;border-radius:15px;background:#F8FBFF;border:1px solid #DCEAFF;min-height:150px}
</style>''', unsafe_allow_html=True)

for k,v in {'df':None,'original_df':None,'history':[],'dataset_versions':[],'file_id':None,'auto_report':None}.items():
    if k not in st.session_state: st.session_state[k]=v

def quality(df):
    if df is None or df.empty:return 0
    cells=max(df.shape[0]*df.shape[1],1)
    score=100-(df.isnull().sum().sum()/cells)*50-(df.duplicated().sum()/max(len(df),1))*30
    return round(max(0,min(100,score)),2)

def outliers(df):
    n=0
    for c in df.select_dtypes(include=np.number).columns:
        s=df[c].dropna()
        if len(s)<4:continue
        q1,q3=s.quantile(.25),s.quantile(.75); i=q3-q1
        n+=int(((s<q1-1.5*i)|(s>q3+1.5*i)).sum())
    return n

def col_info(df):
    return pd.DataFrame([{'Column':c,'Data Type':str(df[c].dtype),'Missing Values':int(df[c].isna().sum()),'Unique Values':int(df[c].nunique())} for c in df.columns])

def report(df):
    return {'Rows':len(df),'Columns':len(df.columns),'Missing Values':int(df.isna().sum().sum()),'Duplicate Rows':int(df.duplicated().sum()),'Outliers':outliers(df),'Empty Columns':sum(df[c].isna().all() for c in df.columns),'Constant Columns':sum(df[c].nunique(dropna=False)<=1 for c in df.columns),'Quality Score':quality(df)}

def save(action,new):
    st.session_state.history.append(action); st.session_state.dataset_versions.append(new.copy())

def undo():
    if len(st.session_state.dataset_versions)>1:
        st.session_state.dataset_versions.pop(); st.session_state.history.pop(); st.session_state.df=st.session_state.dataset_versions[-1].copy(); return True
    return False

def excel(df):
    b=BytesIO()
    with pd.ExcelWriter(b,engine='xlsxwriter') as w: df.to_excel(w,index=False,sheet_name='Cleaned Data')
    return b.getvalue()

def auto_clean(df,dups,missing,text,empty,outs):
    x=df.copy(); actions=[]
    if dups:
        n=int(x.duplicated().sum())
        if n:x=x.drop_duplicates();actions.append(f'Removed {n} duplicate rows')
    if missing:
        for c in x.columns:
            if not x[c].isna().any():continue
            if pd.api.types.is_numeric_dtype(x[c]):
                v=x[c].median(); v=0 if pd.isna(v) else v; x[c]=x[c].fillna(v); actions.append(f"Filled missing values in '{c}' with median")
            else:
                m=x[c].mode(); v=m.iloc[0] if not m.empty else 'Unknown'; x[c]=x[c].fillna(v); actions.append(f"Filled missing values in '{c}' with mode")
    if text:
        cs=x.select_dtypes(include=['object','string']).columns
        for c in cs:x[c]=x[c].astype('string').str.strip()
        if len(cs):actions.append(f'Trimmed spaces from {len(cs)} text columns')
    if empty:
        cs=[c for c in x.columns if x[c].isna().all()]
        if cs:x=x.drop(columns=cs);actions.append(f'Removed {len(cs)} empty columns')
    if outs:
        keep=pd.Series(True,index=x.index)
        for c in x.select_dtypes(include=np.number).columns:
            s=x[c];q1,q3=s.quantile(.25),s.quantile(.75);i=q3-q1;keep&=((s>=q1-1.5*i)&(s<=q3+1.5*i)).fillna(True)
        n=int((~keep).sum());x=x.loc[keep]
        if n:actions.append(f'Removed {n} outlier rows')
    return x,actions

def gemini_client():
    if genai is None:return None,'Install Gemini SDK with: pip install -U google-genai'
    key=None
    try:key=st.secrets['GEMINI_API_KEY']
    except Exception:pass
    if not key:return None,'Gemini API key not configured. Add GEMINI_API_KEY to .streamlit/secrets.toml or Streamlit Cloud Secrets.'
    try:return genai.Client(api_key=key),None
    except Exception as e:return None,f'Gemini initialization error: {e}'

def gemini_analysis(df,question):
    client,err=gemini_client()
    if err:return None,err
    summary=[]
    for c in df.columns:
        summary.append({'column':str(c),'dtype':str(df[c].dtype),'missing':int(df[c].isna().sum()),'missing_percent':round(float(df[c].isna().mean()*100),2),'unique':int(df[c].nunique(dropna=True)),'sample':df[c].dropna().astype(str).head(5).tolist()})
    prompt=f'''You are the data-quality assistant for CleanData AI. Analyze only this dataset summary; do not invent facts. Dataset rows={len(df)}, columns={len(df.columns)}, duplicates={int(df.duplicated().sum())}, outliers={outliers(df)}, quality_score={quality(df)}. Column details={summary}. Give: 1 Overall assessment, 2 Missing-value recommendations, 3 Duplicate recommendations, 4 Outlier recommendations, 5 Data-type recommendations, 6 Text/categorical recommendations, 7 Priority action plan. Explain briefly why each recommendation is appropriate. Do not say an action was performed. User question: {question}'''
    try:
        r=client.models.generate_content(model='gemini-2.5-flash',contents=prompt)
        return r.text,None
    except Exception as e:return None,f'Gemini API error: {e}'

with st.sidebar:
    st.markdown("<h1 style='color:#1557C0;'>🧹 CleanData AI</h1>",unsafe_allow_html=True)
    st.write('Interactive Data Cleaning & Visualization')
    st.markdown('---')
    f=st.file_uploader('📂 Upload Dataset',type=['csv','xlsx','xls'])
    if f is not None and st.session_state.file_id!=(f.name,f.size):
        try:
            d=pd.read_csv(f) if f.name.lower().endswith('.csv') else pd.read_excel(f)
            st.session_state.df=d.copy();st.session_state.original_df=d.copy();st.session_state.history=['Initial Dataset'];st.session_state.dataset_versions=[d.copy()];st.session_state.file_id=(f.name,f.size);st.session_state.auto_report=None;st.success('Dataset uploaded successfully!')
        except Exception as e:st.error(f'Error reading file: {e}')
    menu=st.radio('Navigation',['🏠 Welcome','Dataset Overview','Missing Values','Duplicates','Data Types','Delete Columns','Outliers','Rename Columns','Text Cleaning','Categorical Encoding','Data Visualization','🤖 AI Cleaning Suggestions','⚡ Automated Cleaning Pipeline','📊 Data Quality Dashboard','📥 Export & Cleaning Report','🔄 History & Undo'])
    if st.session_state.df is not None and st.button('🔄 Reset Dataset',use_container_width=True):
        st.session_state.df=st.session_state.original_df.copy();st.session_state.history=['Initial Dataset'];st.session_state.dataset_versions=[st.session_state.original_df.copy()];st.session_state.auto_report=None;st.rerun()

if menu=='🏠 Welcome':
    st.markdown("<div class='main-title'>Welcome to CleanData AI</div>",unsafe_allow_html=True);st.markdown("<div class='subtitle'>Intelligent Data Cleaning & Visualization Platform</div>",unsafe_allow_html=True)
    a,b=st.columns(2)
    with a:
        st.markdown('## 🧹 Clean Your Data Smarter');st.write('Upload CSV/Excel files, clean data, analyze quality, visualize results and get Gemini-powered recommendations.')
    with b:
        p=Path(__file__).resolve().parent/'assets'/'welcome.jpeg'
        if p.exists():st.image(str(p),use_container_width=True)
        else:st.info('Optional: add assets/welcome.jpeg')
    x,y,z=st.columns(3)
    x.markdown('### 🧹 Automated Cleaning');x.write('Missing values, duplicates, text, empty columns and outliers.')
    y.markdown('### 🤖 Gemini AI');y.write('AI-powered recommendations based on your dataset summary.')
    z.markdown('### 📊 Visualization');z.write('Interactive Plotly charts and correlation analysis.')
    if st.session_state.df is None:st.info('👈 Upload your dataset from the sidebar to get started.')
else:
    if st.session_state.df is None:st.warning('👈 Please upload a CSV or Excel dataset first.');st.stop()
    df=st.session_state.df
    if menu=='Dataset Overview':
        st.markdown("<div class='section-title'>📋 Dataset Overview</div>",unsafe_allow_html=True);a,b,c,d=st.columns(4);a.metric('Rows',len(df));b.metric('Columns',len(df.columns));c.metric('Missing',int(df.isna().sum().sum()));d.metric('Duplicates',int(df.duplicated().sum()));st.dataframe(df.head(100),use_container_width=True);st.dataframe(col_info(df),use_container_width=True)
    elif menu=='Missing Values':
        st.markdown("<div class='section-title'>🔍 Missing Values</div>",unsafe_allow_html=True);st.dataframe(pd.DataFrame({'Column':df.columns,'Missing Values':df.isna().sum().values}),use_container_width=True);cs=[c for c in df.columns if df[c].isna().any()]
        if not cs:st.success('✅ No missing values found!')
        else:
            c=st.selectbox('Select column',cs);m=st.selectbox('Method',['Drop Rows','Mean','Median','Mode','Custom Value']);v=st.text_input('Custom value') if m=='Custom Value' else None
            if st.button('Apply Missing Value Treatment',type='primary'):
                x=df.copy()
                if m=='Drop Rows':x=x.dropna(subset=[c])
                elif m in ['Mean','Median']:
                    if not pd.api.types.is_numeric_dtype(x[c]):st.error(f'{m} requires a numeric column.');st.stop()
                    x[c]=x[c].fillna(x[c].mean() if m=='Mean' else x[c].median())
                elif m=='Mode':
                    q=x[c].mode();x[c]=x[c].fillna(q.iloc[0] if not q.empty else 'Unknown')
                else:x[c]=x[c].fillna(v)
                st.session_state.df=x;save(f'Missing values: {m} - {c}',x);st.rerun()
    elif menu=='Duplicates':
        n=int(df.duplicated().sum());st.markdown("<div class='section-title'>🧾 Duplicates</div>",unsafe_allow_html=True);st.metric('Duplicate Rows',n)
        if n:
            st.dataframe(df[df.duplicated(keep=False)],use_container_width=True)
            if st.button('Remove Duplicate Rows',type='primary'):
                x=df.drop_duplicates();st.session_state.df=x;save(f'Removed {n} duplicate rows',x);st.rerun()
        else:st.success('✅ No duplicate rows found.')
    elif menu=='Data Types':
        st.markdown("<div class='section-title'>🔤 Data Types</div>",unsafe_allow_html=True);st.dataframe(col_info(df),use_container_width=True);c=st.selectbox('Column',list(df.columns));t=st.selectbox('Convert to',['int','float','string','datetime'])
        if st.button('Convert Data Type',type='primary'):
            try:
                x=df.copy();x[c]=pd.to_numeric(x[c]).astype('Int64') if t=='int' else pd.to_numeric(x[c]).astype(float) if t=='float' else x[c].astype('string') if t=='string' else pd.to_datetime(x[c])
                st.session_state.df=x;save(f"Converted '{c}' to {t}",x);st.rerun()
            except Exception as e:st.error(f'Conversion failed: {e}')
    elif menu=='Delete Columns':
        st.markdown("<div class='section-title'>🗑️ Delete Columns</div>",unsafe_allow_html=True);cs=st.multiselect('Columns',list(df.columns))
        if st.button('Delete Selected Columns',type='primary'):
            if not cs:st.warning('Select columns.')
            elif len(cs)==len(df.columns):st.error('At least one column must remain.')
            else:x=df.drop(columns=cs);st.session_state.df=x;save(f'Deleted columns: {cs}',x);st.rerun()
    elif menu=='Outliers':
        st.markdown("<div class='section-title'>📌 Outliers</div>",unsafe_allow_html=True);nums=list(df.select_dtypes(include=np.number).columns)
        if not nums:st.info('No numeric columns.')
        else:
            c=st.selectbox('Numeric column',nums);s=df[c].dropna();q1,q3=s.quantile(.25),s.quantile(.75);i=q3-q1;mask=(df[c]<q1-1.5*i)|(df[c]>q3+1.5*i);st.metric('Outliers',int(mask.sum()));st.dataframe(df[mask],use_container_width=True)
            if st.button('Remove Outlier Rows',type='primary'):
                x=df.loc[~mask].copy();st.session_state.df=x;save(f"Removed outliers from '{c}'",x);st.rerun()
    elif menu=='Rename Columns':
        st.markdown("<div class='section-title'>✏️ Rename Columns</div>",unsafe_allow_html=True);c=st.selectbox('Column',list(df.columns));n=st.text_input('New name',str(c))
        if st.button('Rename Column',type='primary'):
            if n.strip() and (n==c or n not in df.columns):x=df.rename(columns={c:n.strip()});st.session_state.df=x;save(f"Renamed '{c}' to '{n.strip()}'",x);st.rerun()
            else:st.error('Invalid or duplicate column name.')
    elif menu=='Text Cleaning':
        st.markdown("<div class='section-title'>📝 Text Cleaning</div>",unsafe_allow_html=True);cs=list(df.select_dtypes(include=['object','string']).columns)
        if not cs:st.info('No text columns.')
        else:
            c=st.selectbox('Text column',cs);op=st.selectbox('Operation',['Strip Spaces','Lowercase','Uppercase','Title Case'])
            if st.button('Apply Text Cleaning',type='primary'):
                x=df.copy();s=x[c].astype('string');s=s.str.strip() if op=='Strip Spaces' else s.str.lower() if op=='Lowercase' else s.str.upper() if op=='Uppercase' else s.str.title();x[c]=s;st.session_state.df=x;save(f'{op}: {c}',x);st.rerun()
    elif menu=='Categorical Encoding':
        st.markdown("<div class='section-title'>🔢 Categorical Encoding</div>",unsafe_allow_html=True);cs=list(df.select_dtypes(include=['object','string','category']).columns)
        if not cs:st.info('No categorical columns.')
        else:
            c=st.selectbox('Categorical column',cs);m=st.selectbox('Encoding',['Label Encoding','One-Hot Encoding'])
            if st.button('Apply Encoding',type='primary'):
                x=df.copy();x[c],_=pd.factorize(x[c]) if m=='Label Encoding' else (None,None)
                if m=='One-Hot Encoding':x=pd.get_dummies(df,columns=[c],dtype=int)
                st.session_state.df=x;save(f'{m}: {c}',x);st.rerun()
    elif menu=='Data Visualization':
        st.markdown("<div class='section-title'>📊 Data Visualization</div>",unsafe_allow_html=True);typ=st.selectbox('Chart',['Histogram','Bar Chart','Scatter Plot','Box Plot','Pie Chart','Correlation Heatmap']);nums=list(df.select_dtypes(include=np.number).columns)
        if typ=='Histogram':
            if not nums:st.info('No numeric columns.')
            else:c=st.selectbox('Column',nums);st.plotly_chart(px.histogram(df,x=c),use_container_width=True)
        elif typ=='Bar Chart':
            c=st.selectbox('Category',list(df.columns));q=df[c].astype(str).value_counts().head(20).reset_index();q.columns=[c,'Count'];st.plotly_chart(px.bar(q,x=c,y='Count'),use_container_width=True)
        elif typ=='Scatter Plot':
            if len(nums)<2:st.info('Need two numeric columns.')
            else:x=st.selectbox('X',nums);y=st.selectbox('Y',nums,index=1);st.plotly_chart(px.scatter(df,x=x,y=y),use_container_width=True)
        elif typ=='Box Plot':
            if not nums:st.info('No numeric columns.')
            else:c=st.selectbox('Column',nums);st.plotly_chart(px.box(df,y=c),use_container_width=True)
        elif typ=='Pie Chart':
            c=st.selectbox('Category',list(df.columns));q=df[c].astype(str).value_counts().head(15).reset_index();q.columns=[c,'Count'];st.plotly_chart(px.pie(q,names=c,values='Count'),use_container_width=True)
        else:
            if len(nums)<2:st.info('Need two numeric columns.')
            else:st.plotly_chart(px.imshow(df[nums].corr(),text_auto=True),use_container_width=True)
    elif menu=='🤖 AI Cleaning Suggestions':
        st.markdown("<div class='section-title'>🤖 Gemini AI Cleaning Suggestions</div>",unsafe_allow_html=True);r=report(df);a,b,c,d=st.columns(4);a.metric('Quality',f"{r['Quality Score']}/100");b.metric('Missing',r['Missing Values']);c.metric('Duplicates',r['Duplicate Rows']);d.metric('Outliers',r['Outliers']);q=st.text_area('Optional question',placeholder='Which cleaning steps should I perform first?')
        if st.button('✨ Analyze with Gemini',type='primary'):
            with st.spinner('Gemini is analyzing your dataset...'):ans,err=gemini_analysis(df,q)
            if err:st.error(err)
            else:st.success('Gemini analysis completed.');st.markdown(ans)
    elif menu=='⚡ Automated Cleaning Pipeline':
        st.markdown("<div class='section-title'>⚡ Automated Cleaning Pipeline</div>",unsafe_allow_html=True);a=st.checkbox('Remove duplicates',True);b=st.checkbox('Handle missing values',True);c=st.checkbox('Clean text spaces',True);d=st.checkbox('Remove empty columns',True);e=st.checkbox('Remove numeric outliers',False)
        if st.button('🚀 Run Automated Cleaning',type='primary'):
            x,acts=auto_clean(df,a,b,c,d,e);st.session_state.df=x;st.session_state.auto_report={'Actions':acts,'Rows':len(x),'Missing':int(x.isna().sum().sum())};save('Automated Cleaning Pipeline',x);st.success('Cleaning completed.');st.write(acts);st.dataframe(x.head(100),use_container_width=True)
    elif menu=='📊 Data Quality Dashboard':
        st.markdown("<div class='section-title'>📊 Data Quality Dashboard</div>",unsafe_allow_html=True);r=report(df);a,b,c,d,e=st.columns(5);a.metric('Quality',f"{r['Quality Score']}/100");b.metric('Rows',r['Rows']);c.metric('Missing',r['Missing Values']);d.metric('Duplicates',r['Duplicate Rows']);e.metric('Outliers',r['Outliers']);st.dataframe(col_info(df),use_container_width=True)
    elif menu=='📥 Export & Cleaning Report':
        st.markdown("<div class='section-title'>📥 Export & Cleaning Report</div>",unsafe_allow_html=True);r=report(df);st.json(r);st.download_button('⬇️ Download CSV',df.to_csv(index=False).encode(),file_name='cleaned_dataset.csv',mime='text/csv');st.download_button('⬇️ Download Excel',excel(df),file_name='cleaned_dataset.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    elif menu=='🔄 History & Undo':
        st.markdown("<div class='section-title'>🔄 History & Undo</div>",unsafe_allow_html=True)
        for i,a in enumerate(st.session_state.history,1):st.write(f'{i}. {a}')
        if st.button('↩️ Undo Last Action',type='primary'):
            if undo():st.success('Last action undone.');st.rerun()
            else:st.info('Nothing to undo.')

st.markdown('---')
st.markdown("<p style='text-align:center;color:#777;'>CleanData AI • Python • Streamlit • Pandas • NumPy • Plotly • Gemini API</p>",unsafe_allow_html=True)
