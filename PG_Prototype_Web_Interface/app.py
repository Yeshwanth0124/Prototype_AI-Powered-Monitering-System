from flask import *
import cv2
from facenet_pytorch import InceptionResnetV1,MTCNN
import torch
from scipy.spatial.distance import euclidean
app = Flask(__name__)
@app.route('/')
@app.route('/home')
def home():
    return render_template("home.html")
@app.route('/login')
def login():
    return render_template("Login.html")
@app.route("/cameras")
def video():
    def generate_frames():
        camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        frame_count = 0
        while True:
            success,frame = camera.read()
            if not success:
                break
            else:
                frame_count+=1
                if frame_count %3!=0:
                    continue
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                mtcnn = MTCNN(keep_all=True,device=device)
                facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
                with torch.no_grad():
                    box,prob = mtcnn.detect(frame)
                    face = frame[y1:y2,x1:x2]
                frame = cv2.resize(frame,(640,480))
                ret,buffer = cv2.imencode(".jpg",frame)
                frame = buffer.tobytes()
                yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')
        camera.release()
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == "__main__":
    app.run(debug=True)