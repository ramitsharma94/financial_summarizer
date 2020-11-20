
from flask import Flask, render_template, url_for, request, redirect, jsonify, Response
# from flask_sqlalchemy import SQLAlchemy
import flask_excel as excel
from datetime import datetime
import pandas as pd
from werkzeug.utils import secure_filename
from flask_uploads import UploadSet, configure_uploads, DOCUMENTS, IMAGES

import pandas as pd
import torch
import time

from models_summarizer import predict_summaries, scrape_web_data
from models_QnA import answer



app = Flask(__name__)
docs = UploadSet('datafiles', DOCUMENTS)
app.config['UPLOADED_DATAFILES_DEST'] = 'static/uploads'
configure_uploads(app, docs)

@app.route("/", methods = ['GET', "POST"])
def index():
    if(request.method == "POST"):
        option = request.form["links"]
        
        if(option == "single_link"):
            try:    
                link = request.form["Enter_Link"]
                input_text = scrape_web_data(link)
                df2 = pd.DataFrame()
                df2["input_text"] = [input_text] 
                df2["Predicted_Summaries"] = [predict_summaries(input_text)]
                df2.drop("input_text", axis =1, inplace = True)
            
                sub_q = "Who is the subject?"
                obj_q = "Who is the object?"

                df2['Subject_Predicted'] = df2['Predicted_Summaries'].apply(lambda x: answer(x, sub_q))
                df2['Object_Predicted'] = df2['Predicted_Summaries'].apply(lambda x: answer(x, obj_q))

                html = df2.to_html() 
                text_file = open("./templates/results.html", "w", encoding = "utf8") 
                text_file.write(html) 
                text_file.close() 
                return render_template("results.html")
            except:
                return "Please enter the correct link!"
        else:
            try:
                filename = request.files['file']
                data = pd.read_excel(filename)
                links = data["Link"]
                dict_links = {}
                for link in links:
                    if(link not in dict_links):
                        dict_links[link] = scrape_web_data(link)
                    else:
                        dict_links[link] = 0
                
                df2 = pd.DataFrame()
                df2["input_text"] = [v for k, v in dict_links.items()]
                print("Web Scraping Done. Prediction Start!")
                summ = []
                for i, text in enumerate(df2["input_text"]):
                    summ.append(predict_summaries(text))
                    print("Done: {}".format(i))
                df2["Predicted_Summaries"] = summ
                df2.drop("input_text", axis =1, inplace = True)
                sub_q = "Who is the subject?"
                obj_q = "Who is the object?"

                df2['Subject_Predicted'] = df2['Predicted_Summaries'].apply(lambda x: answer(x, sub_q))
                df2['Object_Predicted'] = df2['Predicted_Summaries'].apply(lambda x: answer(x, obj_q))

                ## Everything will be written to a html file
                html = df2.to_html() 
                text_file = open("./templates/results.html", "w", encoding = "utf8") 
                text_file.write(html) 
                text_file.close() 
                return render_template("results.html")
            except:
                return "Either the input link is incorrect or the column name is incorrect!"
    else:
        return render_template("index.html")


if __name__=="__main__":
    app.run(debug = True)
