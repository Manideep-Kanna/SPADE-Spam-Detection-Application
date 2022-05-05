
from ML import model
import streamlit as st
from DP import *
import matplotlib.pyplot as plt
import seaborn as sns
inputs=[0,1]
@st.cache()
def create_model():
    mode=model()
    return mode
col1,col2,col3,col4,col5=st.columns(5)
with col3:
    st.title("Spade")
st.write('welcome to Spade...')
st.write('A Spam Detection algorithm based on Machine Learning and Natural Language Processing')

text=st.text_area('please provide email/text you wish to classify',height=400,placeholder='type/paste more than 50 characters here')
file=st.file_uploader("please upload file with your text.. (only .txt format supported")

if len(text)>20:
    inputs[0]=1
if file is None:
    inputs[1]=0
if inputs.count(1)>1:
    st.error('multiple inputs given please select only one option')
else:
    if inputs[0]==1:
        e=text
        given_email = e
    if inputs[1]==1:
        f=open(file,mode='r')
        e=f.read()
        given_email = e



predictions=[]
probs=[]

col1,col2,col3,col4,col5=st.columns(5)
with col3:
    clean_button = st.button('Detect')
st.caption("In case of a warning it's probably related to caching of your browser")
st.caption("please hit the detect button again....")

if clean_button:
    if inputs.count(0)>1:
        st.error('No input given please try after giving the input')
    else:
        with st.spinner('Please wait while the model is running....'):
            mode = create_model()
        given_email,n=clean(given_email)
        vector = mode.get_vector(given_email)
        predictions.append(mode.get_prediction(vector))
        probs.append(mode.get_probabilities(vector))
        col1, col2, col3 = st.columns(3)
        with col2:
            st.header(f"{predictions[0]}")
        st.write('here are some insights into your text')
        probs_pos = [i[1] for i in probs[0]]
        probs_neg = [i[0] for i in probs[0]]
        if predictions[0] == 'Spam':
            # st.caption(str(probs_pos))
            plot_values = probs_pos
        else:
            # st.caption(str(probs_neg))
            plot_values = probs_neg
        col1,col2=st.columns(2)
        #with col1:

        with col1:

            fig, ax = plt.subplots()
            sns.barplot(x=[0, 1, 2, 3, 4], y=plot_values, palette='GnBu')

            plt.title('probabilities')
            plt.xlabel('model')
            plt.ylabel('probability')

            plt.xticks([0, 1, 2, 3, 4], labels=['NB', 'LR', 'RF', 'KNN', 'SVM'])
            st.pyplot(fig)
            st.subheader(f"accuracy - {round(max(plot_values), 2)}")
        with col2:
            entities,explanation=ents(given_email)
            if entities != 'no' and explanation!='no':
                st.subheader('Named Entities')
                st.write(entities)
                st.caption('The explanations for above tags are provied below')
                st.write(explanation)
            else:
                st.subheader('Named Entities not found')






