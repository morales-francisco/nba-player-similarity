#Base Python
FROM python:3.11

#Expose port 8501 for app to be run
EXPOSE 8501

#Set Working Directory
WORKDIR /app

#Copy needed files
COPY requirements.txt /app/requirements.txt
COPY app.py /app/app.py
COPY functions.py /app/functions.py
COPY parameters.py /app/parameters.py
COPY data /app/data

#Install dependencies
RUN pip3 install -r requirements.txt


#Run Streamlit App
CMD streamlit run app.py