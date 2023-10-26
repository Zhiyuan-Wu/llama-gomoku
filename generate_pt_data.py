import json
import random
from utils import action2str

action2strl = lambda y: [action2str(x) for x in y]

def quantity_check():
    data = []
    for i in range(12):
        with open(f"data/selfplay/selfplay_data_{i}.json", 'r') as fin:
            for line in fin:
                _data = json.loads(line)
                data.append( ''.join(action2strl(_data["move_history"])) )

    print(f"total num {len(data)}, valid num {len(set(data))}, ratio {len(set(data))/len(data)}.")

quantity_check()

def template_suggest_move(board, action, pv=None):
    instruction = ["Suggest a possible move.",
                   "Propose a potential next step in the game.",
                   "Recommend an action for the upcoming turn.",
                   "What could be a viable move at this point?",
                   "Share your thoughts on a feasible play right now.",
                   "Advise on a strategy for the next move.",
                   "Offer a suggestion for how to proceed.",
                   "Can you indicate a possible course of action?",
                   "Present an idea for the next play.",
                   "Contemplate and suggest a move.",
                   "What is your recommendation for the forthcoming play?",
                   "Can you advise on a strategic move at this juncture?",
                   "Propose a judicious next step in the current situation.",
                   "Offer your insight on an advantageous move now.",
                   "Suggest a tactical play for the next turn.",
                   "Provide guidance on a potential next move.",
                   "Share a possible action for the ensuing turn.",
                   "What do you think would be a smart move right now?",
                   "Propose an intelligent strategy for the upcoming play.",
                   "Advise on the best course of action for the next move.",
                   ]
    answers     = ["A good option for the next move could be <SuggestMove>.",
                   "Considering the current board, <SuggestMove> seems like a viable move.",
                   "You might want to try <SuggestMove> in this situation.",
                   "A strategic move right now could be <SuggestMove>.",
                   "Based on the game's progression, <SuggestMove> is a recommended move.",
                   "Taking the current positions into account, you could go for <SuggestMove>.",
                   "Given the circumstances, <SuggestMove> could be a wise choice for your next move.",
                   "To gain an advantage, you might consider moving to <SuggestMove>.",
                   "A potential next step to enhance your position could be <SuggestMove>.",
                   "In this scenario, making a move to <SuggestMove> could be beneficial.",
                   "You might find success with the move <SuggestMove> in this situation.",
                   "A strategic option to consider would be <SuggestMove>.",
                   "To advance your position, think about making the move <SuggestMove>.",
                   "Given the current board, <SuggestMove> appears to be a strong move.",
                   "To gain momentum, you could try the move <SuggestMove>.",
                   "Contemplating the present scenario, <SuggestMove> could be a beneficial move.",
                   "Looking at the board, I would suggest the move <SuggestMove>.",
                   "A potential move that stands out is <SuggestMove>.",
                   "In this position, the move <SuggestMove> could be advantageous.",
                   "Given your current standing, <SuggestMove> seems like a solid choice.",
                   ]
    answers_pv  = ["If you move to <SuggestMove>, a possible counter-move could be <FutureMoves>.",
                   "A recommended move is <SuggestMove>, which might lead to the following positions: <FutureMoves>.",
                   "Choosing <SuggestMove> could pave the way for future possibilities such as <FutureMoves>.",
                   "Opting for <SuggestMove> now could result in subsequent moves like <FutureMoves>.",
                   "A strategic play would be <SuggestMove>, setting the stage for moves such as <FutureMoves>.",
                   "The move <SuggestMove> could open up opportunities for <FutureMoves> in the future turns.",
                   "By selecting <SuggestMove>, you might encounter responses like <FutureMoves>.",
                   "If you decide on <SuggestMove>, be prepared for potential moves such as <FutureMoves>.",
                   "A forward-thinking move would be <SuggestMove>, which could lead to <FutureMoves>.",
                   "Go for <SuggestMove> to potentially set up for moves like <FutureMoves> in the ensuing turns.",
                   "Making the move to <SuggestMove> could strategically lead to <FutureMoves> in subsequent turns.",
                   "By choosing <SuggestMove>, you set yourself up for potential plays such as <FutureMoves>.",
                   "A proactive move would be <SuggestMove>, which could provoke reactions like <FutureMoves>.",
                   "If you opt for <SuggestMove>, it might open possibilities for moves like <FutureMoves>.",
                   "Considering the game, <SuggestMove> is a recommended move, potentially leading to <FutureMoves>.",
                   "Going for <SuggestMove> could be wise, as it might set up for <FutureMoves>.",
                   "A calculated move at this point could be <SuggestMove>, anticipating <FutureMoves> in return.",
                   "If you play <SuggestMove>, you may find the game progressing towards <FutureMoves>.",
                   "Choosing <SuggestMove> now could strategically lead to opportunities like <FutureMoves>.",
                   "A thought-out move like <SuggestMove> could pave the way for future scenarios including <FutureMoves>.",
                   ]
    inst = random.choice(instruction)
    if pv is None:
        ans = random.choice(answers)
        ans.replace("<SuggestMove>", action2str(action))
    else:
        ans = random.choice(answers_pv)
        ans.replace("<SuggestMove>", action2str(action))
        ans.replace("<FutureMoves>", '->'.join(action2strl(pv)))
    if random.random()>0.5:
        caption = f"The current game board is <board>.\n{inst}\n{ans}\n\n"
    else:
        caption = f"{inst}\nThe current game board is <board>.\n{ans}\n\n"
    
    return {"board": board, caption: caption}


def template_describe_status():
    instruction = ["Provide a description of the current state of the board.",
                   "Describe the situation of current board.",
                   "Can you detail the current configuration of the game board.",
                   "What is the current layout of the pieces on the board?",
                   "Please characterize the present condition of the board.",
                   "How are the pieces currently arranged on the game board?",
                   "Offer an overview of the board’s current state.",
                   "Summarize the current positioning of the pieces in the game.",
                   "Explain the present setup of the board in the game.",
                   "Give a depiction of how the board looks right now.",
                   "Illustrate the current circumstances on the board.",
                   "Outline the present state of play on the board.",
                   "Depict the current standings in the game based on the board.",
                   "Narrate the ongoing scenario on the game board.",
                   "Share insights on the board’s present setup.",
                   "Discuss the prevailing conditions of the game board.",
                   "Provide a snapshot of the current game situation on the board.",
                   "Evaluate the key areas of focus on the current board.",
                   "Analyze the interest points of the current game board.",
                   "Examine the critical positions on the game board at this moment.",
                   "Assess the strategic points of interest in the current game.",
                   "Explore the pivotal areas on the board right now.",
                   "Investigate the notable aspects of the current board setup.",
                   "Scrutinize the important positions in the ongoing game.",
                   "Dissect the crucial elements of the current board configuration.",
                   "Review the significant parts of the game board as it stands.",
                   "Delve into the areas of importance on the current game board.",
                   "Identify and analyze the high-stakes regions of the board in the current state.",
                   ]