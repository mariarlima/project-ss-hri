# Robotic Conversational AI Engagement for Cognitive Disorders

As the global population ages, the demand for accessible, scalable tools to support cognitive health and detect early signs of cognitive decline is growing. This project investigates the use of verbal human-robot interaction (HRI) and machine learning analysis of speech data as a potential pathway for cognitive engagement and screening. Leveraging recent advances in large language models (LLMs), speech technology, and social robotics, the system is designed to function in real-world, home-based settings.

## 🎯 Learning Outcomes

- Design LLM-powered interactions for verbal human-robot engagement based in the context of an interactive cognitive task (based on clinically validated cognitive tools).
- Program and integrate robot behaviours using a 3D-printed robot platform for multimodal human-robot engagement.
- Implement a machine learning pipeline to analyse cognitive state from speech and language features.
- Discuss the role of conversational AI for accessible cognitive support and early screening in real-world home settings.

## 🛠️ Project Tasks

### 1. Verbal Interaction

- LLM prompt engineering of interactive cognitive task.
- TTS/STT integration to synthesize robot speech and transcribe user speech for two-way verbal interaction.

### 2. Multimodal HRI Integration

- Implement robot motion sequences/gestures (e.g., nodding).
- Synchronise robot motion with speech duration.

#### 🤖 Robotic Platform 
This project will use the updated version of the inexpensive Blossom robot, an [open-source](https://github.com/interaction-lab/Blossom-Controller), 3D-printed platform with a gray crocheted exterior, to create a simple and engaging appearance.
Idle motions will be implemented as actuations for each of Blossom’s four motors to manually selected goal positions. 
These include different sequences (of customised duration) of sighing, posture sways, and gaze shifts.

<div align="center">
  <img src="./images/robotic-platform.png" alt="Blossom" width="400"/>
  <p><em>Blossom robot platform.</em></p>
</div>


### 3. ML Cognitive Analysis from Speech

- Automatic transcription and speech diarisation.
- Feature extraction.
- Pretrained model evaluation.
- Discuss limitations and need for longitudinal datasets for prognostic assessment.

### 4. Real-World Deployment Considerations

- Discuss design and ethical considerations for deploying conversational AI in real-world residential and clinical settings.

## ⚙️ Setup & dependencies
Clone the repository
```
git clone <repo_url>
cd <project_directory>
```

Create your virtual environment and install dependencies
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🚀 Get started
Check the proposed tasks and follow the step-by-step instructions in the files:
- [📄 `1_verbal_hri.ipynb`](./1_verbal_hri.ipynb)
- [📄 `2_sar_integration.ipynb`](./2_sar_integration.ipynb)
- [📄 `3_ml_cog_analysis.ipynb`](./3_ml_cog_analysis.ipynb)