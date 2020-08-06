from flask import Flask, render_template, flash, request,url_for
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import pandas as pd
from werkzeug.utils import secure_filename
#from werkzeug.contrib.cache import SimpleCache
#cache = SimpleCache()

pd.options.display.float_format = '{:,.2f}%'.format

# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

class ReusableForm(Form):
    @app.route('/')
    def student():        
        return render_template('summary.html')
    
    @app.route('/search',methods = ['POST', 'GET'])
    def home():
        if request.method == 'POST':          
            
            return render_template("login-page.html")
        
    @app.route('/route1',methods = ['POST', 'GET'])
    def route1():
        print('Here')
        if request.method == 'POST':          
            return render_template("predict.html")

    @app.route('/result',methods = ['POST', 'GET'])
    def result():
        if request.method == 'POST':
            result = request.form
            if result['UserId'] == 'admin@caps.com' and result['Password'] == 'admin': 
                
                
                return render_template("summary.html")
            else:
                error = "UserId or Password is incorrect. Please try again."
                return render_template("login-page.html",error = error)
            
    @app.route('/searchTable',methods = ['POST', 'GET'])
    def sTable():
        if request.method == 'POST':
            result = request.form
            f = request.files['upImg']
            f.save(secure_filename(f.filename))
            print('Here')
           
            
                
if __name__ == "__main__":
    app.run(debug=True)
