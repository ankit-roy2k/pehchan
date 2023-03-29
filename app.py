from flask import Flask, render_template, redirect, request, flash
from werkzeug.utils import secure_filename
import os
from attendance import main, stop, playsound

UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'gif']
key = os.urandom(24)

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = key

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return redirect('/home')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/start')
def start_attendance():
    playsound('start.mp3')
    main()
    return redirect('/')

@app.route('/stop')
def stopAttendance():
    stop()
    return redirect('/attendance')

@app.route('/attendance', methods=['GET', 'POST'])
def attendance():
    if request.method == 'POST':
        form = request.form
        name = form['name']
        registration = form['registration']
        semester = form['semester']
        branch = form['branch']
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                ext = filename.rsplit('.', 1)[1].lower()
                strname = registration + '_' + name + '_' + semester + '_' + branch + "." +ext
                present_files = os.listdir('static\\uploads')
                if strname not in present_files:
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'],registration + '_' + name + '_' + semester + '_' + branch + "." +ext))
                    flash('Student successfully added!')
                else:
                    flash('Student already present in the system!')
                return redirect('/attendance')
    return render_template('attendance.html')

@app.route('/alertness')
def alertness():
    return render_template('alertness.html')

@app.errorhandler(404)
def error404(e):
    print(e)
    return '404 Page not found'

if __name__ == '__main__':
    app.run(debug=True)