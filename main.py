from flask import Flask
from app import views

app = Flask(__name__)

# url
app.add_url_rule('/base','base', views.base)
app.add_url_rule('/','index', views.index)
app.add_url_rule('/faceapp', 'faceapp', views.faceapp)
app.add_url_rule('/faceapp/photo', 'photo', views.photo, methods=['GET','POST'])
#
if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5001)
