from flask import Flask, render_template, Response
from main_script_opencv import social_distance_detector, motion_detection, people_counter

app = Flask(__name__)

@app.route('/')
def main():
	"""Video streaming home page."""
	return render_template('main.html')

@app.route('/motion_detection.html')
def motion():
	return render_template('motion_detection.html')

@app.route('/socail_detection_detector.html')
def social_distance():
	return render_template('socail_detection_detector.html')

@app.route('/crowd_counter.html')
def crowd():
	return render_template('crowd_counter.html')

@app.route('/video_feed')
def video_feed():
	"""Video streaming route. Put this in the src attribute of an img tag."""
	return Response(gen(motion_detection()))
					# mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
	app.run(debug=True)