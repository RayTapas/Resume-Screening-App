import streamlit as st
import pickle
import re
import nltk

nltk.download('punkt')
nltk.download('stopwords')

clf = pickle.load(open('clf.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))

def cleanResume(resume_text):
    cleanText = re.sub(r'http\S+\s',' ',resume_text)
    cleanText = re.sub(r'RT|cc',' ',cleanText)
    cleanText = re.sub(r'#\S+\s',' ',cleanText)
    cleanText = re.sub(r'@\S+',' ',cleanText)
    cleanText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ',cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ',cleanText)
    cleanText = re.sub('\\s+', ' ', cleanText)
    return cleanText

#web app

def read_uploaded_file(uploaded_file):
    try:
        # Attempt to read the file
        resume_bytes = uploaded_file.read()
        try:
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('utf-8', errors='ignore')
        return resume_text
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def make_prediction(cleaned_resume):
    try:
        input_features = tfidf.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]
        return prediction_id
    except ValueError as e:
        st.error(f"Error in prediction: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error occurred during prediction: {str(e)}")
        return None
    
#Map category ID to category name

category_mapping = {
    6: "Data Science",
    12: "HR",
    0: "Advocate",
    1: "Arts",
    24: "Web Designing",
    16: "Mechanical Engineer",
    22: "Sales",
    14: "Health and fitness",
    5: "Civil Engineer",
    15: "Java Developer",
    4: "Business Analyst",
    21: "SAP Developer",
    2: "Automation Testing",
    11: "Electrical Engineering",
    18: "Operations Manager",
    20: "Python Developer",
    8: "DevOps Engineer",
    17: "Network Security Engineer",
    19: "PMO",
    7: "Database",
    13: "Hadoop",
    10: "ETL Developer",
    9: "DotNet Developer",
    3: "Blockchain",
    23: "Testing",
}


def main():
    try:
        st.title("Resume Screening App")
        uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])

        if uploaded_file is not None:
            file_size = uploaded_file.size
            if file_size == 0:
                st.warning("Uploaded file is empty. Please upload a valid resume.")
            else:
                resume_text = read_uploaded_file(uploaded_file)
                if resume_text:
                    cleaned_resume = cleanResume(resume_text)
                    prediction_id = make_prediction(cleaned_resume)
                    if prediction_id is not None:
                        category_name = category_mapping.get(prediction_id, "Unknown")
                        st.write("Predicted Category:", category_name)
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

    
#python main
if __name__ == "__main__":
    main()