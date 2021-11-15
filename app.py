import io
import os
import glob
from flask import Flask, flash, request, redirect, url_for, render_template
from flask.helpers import send_file
from werkzeug.utils import secure_filename
from classification_models_inference import inference_pipeline
from PIL import Image
import requests
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.')[-1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET","POST"])
def home():
	if request.method == 'POST':
			if 'file' not in request.files:
				if request.form['url'] != '':
					if (allowed_file(request.form['url'])):
						folder = os.path.join(app.instance_path, 'uploads')
						os.makedirs(folder, exist_ok=True)
						files = glob.glob(os.path.join(folder,'*'))
						for f in files:
							os.remove(f)
						return redirect(url_for('predict', name=request.form['url'], url='true'))
					flash('El URL no es válido')
					return redirect(request.url)
				flash('No hay archivo')
				return redirect(request.url)
			file = request.files['file']
			if file.filename == '':
				if request.form['url'] != '':
					if (allowed_file(request.form['url'])):
						folder = os.path.join(app.instance_path, 'uploads')
						os.makedirs(folder, exist_ok=True)
						files = glob.glob(os.path.join(folder,'*'))
						for f in files:
							os.remove(f)
						return redirect(url_for('predict', name=request.form['url'], url='true'))
					flash('El URL no es válido')
					return redirect(request.url)
				flash('No hay archivo seleccionado')
				return redirect(request.url)
			if file:
				if allowed_file(file.filename):
					filename = secure_filename(file.filename)
					folder = os.path.join(app.instance_path, 'uploads')
					os.makedirs(folder, exist_ok=True)
					files = glob.glob(os.path.join(folder,'*'))
					for f in files:
						os.remove(f)
					file.save(os.path.join(folder, secure_filename(filename)))
					return redirect(url_for('predict', name=filename, url='false'))
				else:
					flash('Incorrect format')
					return redirect(request.url)
	return render_template('home.html')

@app.route("/predict")
def predict():
	isUrl = request.args['url']
	if(isUrl == 'true'):
		filepath = request.args['name']
		extension = filepath.rsplit('.')[-1].lower()
		image = filepath
		if(extension == 'png'):
			filename = "download.jpg"
			response = requests.get(filepath)
			im = Image.open(io.BytesIO(response.content)).convert("RGB")
			rgb_im = im.convert('RGB')
			rgb_im.save(os.path.join(os.path.join(app.instance_path, 'uploads'), filename))
			filepath = os.path.join(os.path.join(app.instance_path, 'uploads'), filename)
			image = filename
	else:
		filename = request.args['name']
		extension = filename.rsplit('.', 1)[1].lower()
		filepath = os.path.join(os.path.join(app.instance_path, 'uploads'), filename)
		image = filename
		if(extension == 'png'):
			filename_jpg = filename.rsplit('.', 1)[0] + '.jpg'
			im = Image.open(filepath)
			rgb_im = im.convert('RGB')
			rgb_im.save(os.path.join(os.path.join(app.instance_path, 'uploads'), filename_jpg))
			filepath = os.path.join(os.path.join(app.instance_path, 'uploads'), filename_jpg)
			image = filename_jpg
			
	top_preds = inference_pipeline(net_chosen='squeezenet1_1', 
                   img_path=filepath
                   )
	# Generate plot
	fig = Figure()
	ax = fig.add_subplot(1, 1, 1)
	top_preds.set_index(['class'])['confidence'].head(5).plot(ax=ax, kind='barh', xlabel="Clase")
	ax.invert_yaxis()
	fig.tight_layout()
	# Convert plot to PNG image
	pngImage = io.BytesIO()
	FigureCanvas(fig).print_png(pngImage)
	fig.savefig(os.path.join(os.path.join(app.instance_path, 'uploads'), 'statistics.png'), format='png')

	result = top_preds.head(1)
	return render_template('result.html', value=result, image=image)

@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    if 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'no-store'
    return response

@app.route('/<file_name>')
def returnFile(file_name):
    path_to__file = os.path.join(os.path.join(app.instance_path, 'uploads'), file_name)
    return send_file(
         path_to__file, 
         mimetype="image/jpeg", 
         as_attachment=True, 
         attachment_filename=file_name)

@app.route('/statistics.png')
def returnPlot():
    path_to__file = os.path.join(os.path.join(app.instance_path, 'uploads'), 'statistics.png')
    return send_file(
         path_to__file, 
         mimetype="image/png", 
         as_attachment=True, 
         attachment_filename='statistics.png')

if __name__ == '__main__':
	app.run(debug=False, use_reloader=True)