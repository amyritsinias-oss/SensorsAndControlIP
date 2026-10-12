# mujoco UR3 tutorial

Minimalist tutorial to set up a MuJoCo simulation

## Install Mujoco

First create a virtual environment
```bash
python -m venv .venv
```

Activate it:
```bash
source .venv/bin/activate
```

Install mujoco:
```bash
pip install mujoco
```

## Load the UR3

UR3 files: [https://github.com/SouthColumn76/universal_robots_ur3e](https://github.com/SouthColumn76/universal_robots_ur3e)

```bash
sudo apt install git
git clone git@github.com:SouthColumn76/universal_robots_ur3e.git
```
