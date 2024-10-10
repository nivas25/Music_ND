from flask import Flask, render_template, request, send_file
from audiocraft.models import MusicGen
import torch
import torchaudio
import os
import base64

app = Flask(__name__, static_url_path='/static', static_folder='static')

def load_model():
    model = MusicGen.get_pretrained('facebook/musicgen-small')
    return model

def generate_music_tensors(description, duration):
    print("Description: ", description)
    print("Duration: ", duration)
    model = load_model()

    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=duration
    )

    output = model.generate(
        descriptions=[description],
        progress=True,
        return_tokens=True
    )

    return output[0]

def save_audio(samples):
    sample_rate = 32000
    save_path = os.path.join(app.static_folder, "audio_output")
    assert samples.dim() == 2 or samples.dim() == 3

    samples = samples.detach().cpu()
    if samples.dim() == 2:
        samples = samples[None, ...]

    for idx, audio in enumerate(samples):
        audio_path = os.path.join(save_path, f"audio_{idx}.wav")
        torchaudio.save(audio_path, audio, sample_rate)
    return audio_path  # Moved outside of the loop

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sample')
def samples():
    return render_template('samples.html')


@app.route('/generate_music', methods=['POST'])
def generate_music():
    description = request.form['description']
    duration = int(request.form['duration'])

    music_tensors = generate_music_tensors(description, duration)
    audio_path = save_audio(music_tensors)

    return render_template('audio_player.html', audio_path=audio_path)

@app.route('/download_audio', methods=['GET'])
def download_audio():
    audio_path = request.args.get('audio_path')
    return send_file(audio_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)