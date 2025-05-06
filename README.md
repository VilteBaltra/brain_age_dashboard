
# brain_age_dashboard

**Set up environment**  
```bash
conda create -n brainage-env -c conda-forge python=3.10  
conda activate brainage-env
```

**Install requirements**  
```bash
pip install -r requirements.txt
```

**Run Streamlit app locally**  
```bash
streamlit run build_dashboard_brain_age.py
```

**Run ngrok to expose the Streamlit app**  
```bash
# to install ngrok follow steps detailed here https://dashboard.ngrok.com/get-started/setup/
ngrok http 8501
```

**Get the ngrok temporary public URL**  
```bash
# the URL listed after Forwarding (e.g., http://abcd1234.ngrok.io) is the public link that you can now share with others
Forwarding                    http://abcd1234.ngrok.io -> http://localhost:8501
```



