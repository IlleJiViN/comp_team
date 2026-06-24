from fastapi import FastAPI, UploadFile, File
import shutil
import os
import uvicorn
from fastapi.responses import FileResponse

app = FastAPI()

@app.get("/download/{filename}")
def download_csv(filename: str):
    return FileResponse(f"data/{filename}", media_type="text/csv", filename=filename)

@app.post("/upload")
def upload_npz(file: UploadFile = File(...)):
    os.makedirs("data", exist_ok=True)
    file_location = f"data/{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    # After receiving the file, we can also trigger the DB update!
    # But wait_and_update_v6.py is already doing it or we can just call it here.
    # For now, just save it.
    print(f"Successfully received {file.filename} from Colab!")
    return {"info": f"file '{file.filename}' saved at '{file_location}'"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
