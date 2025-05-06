# brain_age_dashboard

# set up environment
conda create -n brainage-env -c conda-forge python=3.10 
conda activate brainage-env

# install requirements
pip install -r requirements.txt

# run streamlit
streamlit run build_dashboared_brain_age.py