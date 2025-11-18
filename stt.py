import whisper
import json
model = whisper.load_model("large-v2")
result = model.transcribe(audio= ,
                          language= "hi",
                          task="translate",
                          word_timestamps=False)
print(result["text"]) 
with open("ouput.json","w")as f:
    json.dump(f,result)