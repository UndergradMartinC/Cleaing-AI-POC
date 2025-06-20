from flask import Flask, render_template, request, redirect, url_for
import os
from detect import run_detection
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
@app.route('/', methods=['GET', 'POST'])
def index():
    image_url = None
    timestamp = int(time.time())

    if request.method == 'POST':
        before = request.files['before']
        after = request.files['after']

        if before and after:
            before_path = os.path.join(app.config['UPLOAD_FOLDER'], 'before1.jpg')
            after_path = os.path.join(app.config['UPLOAD_FOLDER'], 'after1.jpg')
            before.save(before_path)
            after.save(after_path)

            output_path = 'static/difference.jpg'
            run_detection(before_path, after_path, output_path)

            image_url = output_path

    return render_template('index.html', image_url=image_url, timestamp=timestamp)

if __name__ == '__main__':
    app.run(debug=True)