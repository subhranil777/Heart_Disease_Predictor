from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__, template_folder='.')

model=pickle.load(open('model.pkl','rb'))


def make_prediction(final):
     prediction=model.predict(final)
     if(prediction[0]==0):
          return ('The person does not have a heart disease.')
     elif(prediction[0]==1):
         return  ('The person has low risk  heart disease.')
     elif(prediction[0]==2):
         return ('The person has medium risk heart disease.')
     elif(prediction[0]==3):
         return ('The person has high risk heart disease.')
     else:
         return ('The person has serious heart disease.')

     # return prediction

def get_plot1(data):

        colors = ['blue', 'green']
        values=[data,200]
        labels = ['Patient Cholesterol', 'Normal Cholesterol']
        plt.bar(labels,values,color=colors)
        plt.title('Cholesterol Level')
        plt.xlabel('Patient vs normal level')
        plt.ylabel('Cholesterol (mg/dL)')
        plt.axhline(200, color='red', linestyle='--', label='Normal Cholesterol Level')
        plt.savefig('static/graph1.png')
        plt.close()
     #    x=np.arange(len(data))
     #    y=200
     #    plt.bar(x,y)
     #    plt.xlabel('X Label')
     #    plt.ylabel('Y Label')
     #    plt.title('Mock Graph')
     #    plt.savefig('static/graph.png')
     #    plt.close()


def get_plot2(thalach,age):

        colors = ['blue', 'green']
        values=[thalach,220-age]
        labels = ['Patient heart_rate', 'normal heart_rate']
        plt.bar(labels,values,color=colors)
        plt.title('Maximum Heart Rate')
        plt.xlabel('Patient vs normal level')
        plt.ylabel('Heart_rate(bpm)')
        plt.axhline(220-age, color='red', linestyle='--', label='Maximum safe heart_rate')
        plt.savefig('static/graph2.png')
        plt.close()


def get_plot3(trestbps):

        colors = ['blue', 'green']
        values=[trestbps,120]
        labels = ['Resting Blood Pressure','normal blood pressure']
        plt.bar(labels,values,color=colors)
        plt.title('Resting Blood Pressure')
        plt.xlabel('Patient vs normal level')
        plt.ylabel('Resting Blood Pressure(mm Hg)')
        plt.axhline(120, color='red', linestyle='--', label='Maximum safe limit')
        plt.savefig('static/graph3.png')
        plt.close()


@app.route('/')
def hello_world():
    return render_template('index.html')




@app.route('/predict',methods=['POST','GET'])

def predict():
    chol=int(request.form['chol'])
    thalach=int(request.form['thalach'])
    trestbps=int(request.form['trestbps'])
    age=int(request.form['age'])
    int_features=[float(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=make_prediction(final)
    get_plot1(chol)
    get_plot2(thalach,age)
    get_plot3(trestbps)
    return render_template('plot.html',prediction=prediction)

#     if(prediction[0]==0):
#           return render_template('heart.html',pred='The person does not have a heart disease.')
#     elif(prediction[0]==1):
#          return  render_template('heart.html',pred='The person has low risk  heart disease.')
#     elif(prediction[0]==2):
#          return render_template('heart.html',pred='The person has medium risk heart disease.')
#     elif(prediction[0]==3):
#          return render_template('heart.html',pred='The person has high risk heart disease.')
#     else:
#          return render_template('heart.html',pred='The person has serious heart disease.')

     

    
# @app.route('/predict',methods=['POST','GET'])



if __name__ == '__main__':
     app.run('127.0.0.1',5000,debug=True)