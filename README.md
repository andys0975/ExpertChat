# TalkWithExpert

Demo video of this project:
<video src="demo.mp4"></video>

## Features
1. Streaming response from the expert LLM on Streamlit App.
2. LLM replies mimicking the expert's speech style.
3. LLM replies based on the retrieved content from expert's videos.
4. The reference video links are provided.
5. Pre-made audio responses based on the expert's voice.
6. Print time taken for each response in the console.

- Please note that the app.py is the final version. The dev.ipynb is just a kind of development logs.

## Development Environment
- MacBook Air (M3, 2024)
- macOS Sequoia 15.1
- Python 3.11.10

## Workflow
1. Clone this repository.
```bash
git clone https://github.com/andys0975/TalkWithExpert
```
2. Unzip the `materials.zip` file in the repository directory.
```bash
cd TalkWithExpert
unzip materials.zip
```
3. The repository tree should look like this.
```markdown
TalkWithExpert
├── reference/
│   ├── expert_intro.json
│   ├── Scott_Manley.csv
│   └── Fireship.csv
├── vectors/
│   ├── expert_intro_vectors.npy
│   ├── Scott_Manley_vectors.npy
│   └── Fireship_vectors.npy
├── tests/
│   ├── response_Q1.wav
│   ├── response_Q2.wav
│   └── response_Q3.wav
├── app.py
├── data.py
├── helper.py
├── README.md
├── requirements.txt
└── dev.ipynb
```
4. Make sure the OpenAI API key is set in the environment variable `OPENAI_API_KEY`.
```bash
export OPENAI_API_KEY="your_api_key_here"
```
5. Install the required python packages. (You should consider the device_map for pytorch)
```bash
pip install -r requirements.txt
```
6. Use git lfs to clone the embedding model into preferred path and remember the path for variable `model_path`.
```bash
git lfs install
git clone https://huggingface.co/dunzhang/stella_en_400M_v5
```
7. Run the app with the following command. (You should consider the device_map for pytorch)
```bash
streamlit run app.py "model_path" "device_map"
```