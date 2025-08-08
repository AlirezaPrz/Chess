# ChessAI Bot

A multi‐mode chess AI project featuring:

- **Easy** mode: a Deep Q-Network (DQN) agent trained via OpenAI Gym & TensorFlow  
- **Medium/Hard** modes: classical minimax search with alpha-beta pruning and a phase-aware static evaluation  
- Interactive Pygame interface for human vs. AI play

---

## Features

- **Custom Gym environment** (`ChessEnv.py`) for RL training  
- **Fast static evaluation** tuned for opening development, middlegame tactics and endgame conversion  
- **Adaptive search depth** based on material count (quicker in opening, deeper in endgame)  
- **Pygame GUI** with piece sprites, menus & animations  
- Clear project structure and comprehensive documentation
- Bots in three levels, with the hardest being at 1400 elo

---

## Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/AlirezaPrz/ChessAI.git
   cd ChessAI/Chess
   ```

2. **Create & activate a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate       # macOS/Linux
   venv\Scripts\activate          # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Train the (RL) Bots

```bash
python src/train_model.py
```

This will spin up the Gym environment, train the DQN agent, and save model weights under `weights/`.

### 2. Play against the AI

```bash
python src/chess_main.py
```

* On launch, choose **Easy**, **Medium**, or **Hard** mode.
* Use mouse clicks to move pieces.
* **Z** to undo, **R** to restart, **Q** to quit to menu.

---

## Project Layout

```
Chess/                      
├── images/                 # Piece sprites & menu graphics  
│   ├── set1/  
│   ├── set2/  
│   └── start/  
├── src/                    # Source code  
│   ├── weights/            # Trained DQN model files
│   ├── button.py           # UI button class  
│   ├── chess_engine.py      # Core engine & move generation  
│   ├── chess_env.py         # Gym environment wrapper for RL  
│   ├── minimax_bot.py       # Minimax + evaluation + search  
│   ├── train_model.py       # RL training script for DQN agent  
│   ├── chess_main.py        # Pygame entry-point & menus  
│   └── help.py             # Misc utility functions  
├── requirements.txt        # `pip install` dependency list  
├── CONTRIBUTING.md         # Contribution guidelines  
├── LICENSE                 # MIT License  
└── README.md               # This file  
```

---

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:

1. Forking & cloning
2. Branch naming conventions
3. Coding style checks & pull request process

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

*Happy coding & enjoy your games!*

