import streamlit as st
import pickle
log_model = pickle.load(open('/home/karan/.config/spyder-py3/log_model.pkl','rb'))
rf_model = pickle.load(open('/home/karan/.config/spyder-py3/rf_model.pkl','rb'))
def classify(num):
    if num<0.5:
        return "setosa"
    elif num<1.5:
        return "versicolor"
    else:
        return "virginica"

def main():
    st.title("Iris Prediction")
    html_temp="""
    <div style = "background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Iris Classification</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html= True)
    activities=["Logistic Regression", "Random Forest"]
    option=st.sidebar.selectbox('Which Model Would you like to use?', activities)
    st.subheader(option)
    sl = st.number_input('Select Sepal Length')
    sw = st.number_input('Select Sepal Width')
    pl = st.number_input('Select Petal Length')
    sl = st.number_input('Select Petal Width')
    inputs=[[sl,sw,pl,sl]]
    if st.button("Classify"):
        if option == "Logistic Regression":
            st.success(log_model.predict(inputs))
        else:
            st.success(rf_model.predict(inputs))
            
if __name__== '__main__':
    main()
            
            
    

