
<center>
    <h1>faceawayd</h1>
    A daemon (*technically AI microservice) for human recognition with machine learning
</center>

---

## 1. Requirements

+ Python >=3.13.0
+ A supported CUDA GPU (Pytorch GPU Support)

<i>Running with a CPU may cause some heavy performance penalty.</i>

## 2. Installation

Start by cloning the repository to your system.

```sh
git clone https://github.com/SynopticsKHKT/faceawayd
pip install -r requirements.txt
```

Add your faces to `faces` directory. (JPEG or PNG only)

Modify `mlopts.py` to include the faces:

```py
# The system will automatically load <name>.jpg or <name>.png
# You can use multiple faces & angles to increase accuracy.
known_face_names = [ "name1", "name2", "name3" ]
```

Finally, run your AI Microservice.

```sh
python3 main.py
```

## 3. Why?

+ It provides a cheaper alternative to most AI Camera tracker systems, using ReID & Face Recognition
+ Predicts & provides sustainable solutions to save housing electricity.
