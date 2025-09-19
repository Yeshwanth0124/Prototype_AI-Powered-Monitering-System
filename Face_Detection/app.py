from flask import *
import cv2
import torch
from facenet_pytorch import MTCNN,InceptionResnetV1
import joblib
app = Flask(__name__)
@app.route('/')
def home():
    return render_template("homepage.html")
@app.route('/video')
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
                    for boxes in box:
                        if boxes is None:
                            continue
                        x1,y1,x2,y2 = [int(coord) for coord in boxes]
                    face_tensor = mtcnn(frame)
                    if face_tensor is None:
                        continue
                    if len(face_tensor) == 3:
                        batch_tensor = face_tensor.unsqueeze(0).to(device)
                    else:
                        batch_tensor = face_tensor.to(device)
                    live_embedding = facenet(batch_tensor).detach().cpu().squeeze(0).reshape(1,-1)
                    model = joblib.load('KNC_Model.joblib_update')
                    known_faces = model._fit_X
                    label = model.predict([live_embedding.flatten()])
                    dist,idx = model.kneighbors(live_embedding,n_neighbors=1)
                    if dist[0][0]<0.8:
                            cv2.putText(frame,str(label),(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0))
                    else:
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0))
                frame = cv2.resize(frame,(640,480))
                ret,buffer = cv2.imencode(".jpg",frame)
                frame = buffer.tobytes()
                yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')
        camera.release()
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == "__main__":
    app.run(debug=True)