import json
import random
import numpy as np
import tqdm
import torch.multiprocessing as mp
from utils import *

def game_repetation_check():
    data = []
    for i in range(12):
        with open(f"data/selfplay/selfplay_data_{i}.json", 'r') as fin:
            for line in fin:
                _data = json.loads(line)
                data.append( ''.join(action2strl(_data["move_history"])) )

    print(f"total num {len(data)}, valid num {len(set(data))}, ratio {len(set(data))/len(data)}.")

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
        ans = ans.replace("<SuggestMove>", action2str(action))
    else:
        ans = random.choice(answers_pv)
        ans = ans.replace("<SuggestMove>", action2str(action))
        ans = ans.replace("<FutureMoves>", '->'.join(action2strl(pv)))
    if random.random()>0.5:
        prompt = f"The current game board is <board>.\n{inst}"
        completion = f"{ans}"
    else:
        prompt = f"{inst}\nThe current game board is <board>."
        completion = f"{ans}"
    
    return {"canonicalBoard": board, "prompt": prompt, "completion": completion}


def template_describe_status(board):
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
                   "Provide an assessment of the current conditions on the game board.",
                   "Elaborate on the status of the game based on the board’s current arrangement.",
                   "What can you tell about the ongoing game from the current board configuration?",
                   "Give your perspective on the current situation in the game based on the board.",
                   "How would you characterize the state of play in the game at this moment?",
                   "Provide an analysis of the current game scenario as reflected by the board.",
                   "Describe the board’s current setup and its implications for the game.",
                   "Share your observations on the current game status as seen on the board.",
                   "What does the arrangement of pieces on the board tell us about the current game?",
                   "Elaborate on the strategic implications of the current board configuration.",
                   "Provide a rundown of the current game situation based on the board setup.",
                   "What insights can you gather about the game from the current board arrangement?",
                   "How is the game progressing based on the current layout of the pieces?",
                   "Can you provide a briefing on the current state of affairs in the game?",
                   "Describe the key elements of the current game based on the board’s layout.",
                   "What does the current board configuration reveal about the state of the game?",
                   "Share your thoughts on the game’s progress given the current board setup.",
                   "How would you summarize the ongoing game based on the board’s current state?",
                   "Describe the significant aspects of the current game as shown on the board.",
                   "Can you give an analysis of the game’s status based on the current board?",
                   "Can you identify the key positions on the board that are crucial at this moment?"
                   "Which spots on the board should the players pay special attention to right now?",
                   "What are the strategic points of interest for both players in the current state of the game?",
                   "Could you highlight the positions on the board that are of utmost importance currently?",
                   "Where are the critical areas that could potentially determine the outcome of the game?",
                   "What are the pivotal positions on the board that both players need to consider?",
                   "Can you point out the significant spots on the board that might influence the next few moves?",
                   "Which positions on the board are the players likely focusing on at this moment?",
                   "What are the areas of interest that could play a vital role in the current game situation?",
                   "Can you specify the crucial positions on the board that both players should be mindful of?",
                   "Where are the key points on the board that could lead to a significant advantage for either player?",
                   "What positions on the board hold the most strategic value in the current state of play?",
                   "Could you indicate the spots on the board that are currently under contention?",
                   "Which areas on the board are critical for the players to address or capitalize on?",
                   "What are the significant points on the board that might be game-changers at this moment?",
                   "Can you point out the crucial positions that both players are likely contending for?",
                   "Which spots on the board are the most contentious or significant right now?",
                   "What are the strategic areas that the players should focus on in the current situation?",
                   "Can you highlight the points of interest that could potentially shape the course of the game?",
                   "Which positions on the board are crucial for establishing control or gaining an advantage?",
                   ]
    
    answer_single_threat = ["At <Move>, the <Player> can leverage these positions to create <Pattern> patterns, gaining a strategic advantage.",
                            "The positions at <Move> are crucial for the <Player> to establish <Pattern> formations and maintain control.",
                            "By optimally placing a stone at <Move>, the <Player> can form a <Pattern> pattern, keeping the initiative.",
                            "The <Player> has the opportunity to strengthen their position with <Pattern> patterns by playing at <Move>.",
                            "At <Move>, the potential to create <Pattern> patterns is significant, providing the <Player> with strategic options.",
                            "To maintain momentum, the <Player> should consider forming <Pattern> patterns starting from <Move>.",
                            "The positions at <Move> are key for the <Player>, allowing for the creation of strong <Pattern> patterns.",
                            "The <Player> can seize a strategic advantage by forming <Pattern> patterns from <Move>.",
                            "At <Move>, the <Player> has the chance to establish a commanding position with <Pattern> patterns.",
                            "By focusing on <Move>, the <Player> can create <Pattern> formations, maintaining pressure on the opponent.",
                            "The <Player> can enhance their board presence by creating <Pattern> patterns at <Move>.",
                            "Strategic <Pattern> formations initiated from <Move> can give the <Player> a significant advantage.",
                            "At <Move>, the <Player> has the potential to construct powerful <Pattern> patterns, controlling the game's tempo.",
                            "Focusing on <Move> could allow the <Player> to establish formidable <Pattern> formations, maintaining initiative.",
                            "The positions at <Move> are ripe for the <Player> to create <Pattern> patterns, dictating the pace of play.",
                            "By capitalizing on the opportunities at <Move>, the <Player> can form <Pattern> patterns, gaining a strategic edge.",
                            "The <Player> should aim to build <Pattern> patterns starting from <Move> to maintain their momentum.",
                            "Strategically important <Pattern> formations can be achieved by the <Player> through optimal play at <Move>.",
                            "At <Move>, the <Player> has a prime opportunity to create <Pattern> patterns, securing a strong position.",
                            "The <Player> can establish a commanding presence on the board by forming <Pattern> patterns from <Move>.",
                            "At <Move>, the <Player> can leverage these positions to create <Pattern> patterns.",
                            "The <Pattern> patterns accessible from <Move> could be a key focus for the <Player>.",
                            "With a strategic approach at <Move>, forming <Pattern> patterns is within reach for the <Player>.",
                            "The <Player> has the potential to develop <Pattern> patterns starting from <Move>.",
                            "Considering the positions at <Move>, the <Player> could aim to create <Pattern> patterns.",
                            "At <Move>, the opportunity to form <Pattern> patterns is a viable option for the <Player>.",
                            "The <Player> has the potential to navigate from <Move> towards creating <Pattern> patterns.",
                            "From the positions at <Move>, the <Player> can work towards establishing <Pattern> patterns.",
                            "The <Player> could utilize the positions at <Move> to potentially create <Pattern> patterns.",
                            "At <Move>, forming <Pattern> patterns could be a strategic choice for the <Player>.",
                            "The positions at <Move> provide a foundation for the <Player> to build <Pattern> patterns.",
                            "At <Move>, the <Player> has access to positions that could lead to <Pattern> formations.",
                            "Exploring possibilities at <Move> could result in the <Player> creating <Pattern> patterns.",
                            "The <Player> might find opportunities to establish <Pattern> patterns starting from <Move>.",
                            "At <Move>, there are potential pathways for the <Player> to achieve <Pattern> formations.",
                            "The board positions at <Move> hold the promise of <Pattern> patterns for the <Player>.",
                            "Focusing on <Move> could unlock the potential for the <Player> to form <Pattern> patterns.",
                            "The <Player> has a chance to maneuver into <Pattern> patterns from the positions at <Move>.",
                            "At <Move>, the <Player> can explore options to create <Pattern> patterns.",
                            "Potential <Pattern> formations at <Move> are worth considering for the <Player>.",
                            "The <Player> can aim to transition from <Move> to <Pattern> patterns as a strategic move.",
                            "Exploring the possibilities at <Move> could lead the <Player> to <Pattern> formations.",
                            "At <Move>, the <Player> has the opportunity to work towards creating <Pattern> patterns.",
                            "The positions available at <Move> could facilitate the formation of <Pattern> patterns for the <Player>.",
                            "With a focus on <Move>, the <Player> can potentially navigate towards <Pattern> patterns.",
                            "The <Player> might capitalize on the positions at <Move> to create <Pattern> patterns.",
                            "At <Move>, there is potential for the <Player> to develop <Pattern> formations.",
                            "The <Player> can utilize the positions at <Move> as a stepping stone to create <Pattern> patterns.",
                            "With strategic play at <Move>, the <Player> could achieve <Pattern> formations.",
                            "The board positions at <Move> could be key for the <Player> in forming <Pattern> patterns.",
                            ]
    
    answer_double_threat = ["The <Player> has a chance to dominate the game by creating <Pattern> patterns at <Move>.",
                            "At <Move>, forming <Pattern> patterns could be a game-changing move for the <Player>.",
                            "The positions at <Move> offer the <Player> a powerful opportunity to establish <Pattern> patterns.",
                            "With the potential to create <Pattern> patterns at <Move>, the <Player> could secure a significant advantage.",
                            "The <Player> can exert immense pressure on the opponent by aiming for <Pattern> patterns at <Move>.",
                            "A strategic play at <Move> could lead the <Player> to formidable <Pattern> patterns, shifting the game in their favor.",
                            "The <Player> has the opportunity to create a decisive advantage with <Pattern> patterns starting from <Move>.",
                            "By focusing on <Move>, the <Player> could forge <Pattern> patterns, paving the way to victory.",
                            "The positions at <Move> are crucial for the <Player>, providing a pathway to <Pattern> patterns and a potential win.",
                            "With a well-planned move to <Move>, the <Player> can create <Pattern> patterns and take control of the game.",
                            "A play at <Move> with an aim for <Pattern> patterns could be a pivotal moment for the <Player>.",
                            "The <Player> has a golden opportunity to change the course of the game with <Pattern> patterns at <Move>.",
                            "By achieving <Pattern> patterns at <Move>, the <Player> could move a step closer to victory.",
                            "The strategic significance of <Move> is amplified by the potential to form game-altering <Pattern> patterns.",
                            "A masterful play at <Move> could see the <Player> creating <Pattern> patterns, a move that might clinch the game.",
                            "The positions at <Move> are ripe for the <Player> to establish <Pattern> patterns, potentially deciding the game's outcome.",
                            "With a chance to create <Pattern> patterns at <Move>, the <Player> could dramatically strengthen their position.",
                            "The <Player> can seize a decisive advantage by navigating towards <Pattern> patterns from <Move>.",
                            "At <Move>, the possibility of forming <Pattern> patterns offers the <Player> a route to dominate.",
                            "A calculated move to <Move> with the aim of creating <Pattern> patterns could be a game-winner for the <Player>.",
                            "The positions at <Move> present the <Player> with a unique opportunity to craft <Pattern> patterns, a potentially winning strategy.",
                            "By focusing on <Move>, the <Player> could unlock the power of <Pattern> patterns, tilting the game in their favor.",
                            "A move to <Move> with an eye on creating <Pattern> patterns could be the <Player>'s key to victory.",
                            "The potential for <Pattern> patterns at <Move> makes it a crucial point of interest for the <Player>, possibly leading to a win.",
                            "At <Move>, the <Player> has the option to create <Pattern> patterns, a move that could prove to be decisive.",
                            "The strategic depth of <Move> is highlighted by the <Player>’s ability to form <Pattern> patterns, a potential game-changer.",
                            "With the ability to craft <Pattern> patterns at <Move>, the <Player> is in a strong position to dictate the game's pace."
                            "The <Player>'s path to victory could be paved by forming <Pattern> patterns starting from <Move>.",
                            "At <Move>, the <Player> can make a significant impact by aiming for <Pattern> patterns, potentially leading to a win.",
                            "A tactical play at <Move> could enable the <Player> to establish <Pattern> patterns, putting them in command of the game.",
                            "The <Player> can capitalize on the positions at <Move> to forge <Pattern> patterns, a maneuver that could be decisive."
                            "By converting the potential at <Move> into <Pattern> patterns, the <Player> could significantly bolster their winning chances.",
                            "The game board is set for the <Player> to make a powerful statement by creating <Pattern> patterns at <Move>.",
                            "With the capability to establish <Pattern> patterns from <Move>, the <Player> is in a prime position to assert dominance.",
                            "The positions at <Move> are a springboard for the <Player> to create <Pattern> patterns and steer the game towards victory.",
                            "A strategic exploitation of <Move> could lead the <Player> to form <Pattern> patterns, tipping the scales in their favor.",
                            "The <Player> has a strategic opportunity to change the game's momentum by creating <Pattern> patterns at <Move>.",
                            "The potential for creating <Pattern> patterns at <Move> places the <Player> in a strong position to challenge for victory.",
                            "With <Move>, the <Player> has the possibility to create <Pattern> patterns, a crucial step towards securing a win.",
                            "A play at <Move> targeting <Pattern> patterns could be the <Player>'s key to unlocking a winning position.",
                            "The positions at <Move> offer the <Player> a pathway to create <Pattern> patterns and gain a decisive advantage.",
                            "A successful formation of <Pattern> patterns starting from <Move> could be a game-winning strategy for the <Player>.",
                            "The <Player>’s chances of winning could be significantly enhanced by creating <Pattern> patterns at <Move>.",
                            "At <Move>, the <Player> can initiate a powerful sequence leading to <Pattern> patterns and a potential win.",
                            "The positions at <Move> are pivotal for the <Player>, with the potential to create <Pattern> patterns and dominate the game.",
                            "A calculated approach to <Move> could enable the <Player> to craft <Pattern> patterns, a move with potential winning implications.,"
                            "By focusing on <Move>, the <Player> could transition into a commanding position through <Pattern> patterns.",
                            "The opportunity to create <Pattern> patterns at <Move> is a strategic goldmine for the <Player>, potentially leading to victory.",
                            "With the potential for <Pattern> patterns at <Move>, the <Player> is in a prime position to take control of the game.",
                            "The <Player> can turn the tide of the game by leveraging the positions at <Move> to create <Pattern> patterns.",
                            ]
    
    answer_five          = ["The <Player> can secure a win with a five-in-a-row formation at <Move>.",
                            "A victory is within reach for the <Player> by forming five in a row at <Move>.",
                            "The <Player> has the opportunity to clinch the game with a five-in-a-row at <Move>.",
                            "With a strategic move to <Move>, the <Player> can achieve a winning five-in-a-row.",
                            "The game could be decided if the <Player> forms a five-in-a-row sequence at <Move>.",
                            "Victory is on the horizon for the <Player> with a potential five-in-a-row at <Move>.",
                            "The <Player> is in a prime position to win by creating a five-in-a-row at <Move>.",
                            "A winning move for the <Player> could be to form a five-in-a-row at <Move>.",
                            "The <Player> has a clear path to victory with a possible five-in-a-row formation at <Move>.",
                            "With a strategic placement at <Move>, the <Player> can achieve a five-in-a-row and win the game.",
                            "The positions at <Move> are ripe for the <Player> to form a winning five-in-a-row.",
                            "A decisive move to <Move> could lead the <Player> to a five-in-a-row and secure the win.",
                            "The game board is set for the <Player> to create a five-in-a-row at <Move> and claim victory.",
                            "By focusing on <Move>, the <Player> has the potential to form a winning five-in-a-row.",
                            "The <Player> can capitalize on the opportunity to win by creating a five-in-a-row at <Move>.",
                            "With the potential for a five-in-a-row at <Move>, the <Player> is in a strong position to win.",
                            "The <Player> can aim for a game-winning five-in-a-row at <Move>.",
                            "Victory is in sight for the <Player> with the possibility of a five-in-a-row at <Move>.",
                            "The positions at <Move> provide the <Player> a chance to win with a five-in-a-row.",
                            "A well-placed stone at <Move> could lead the <Player> to a victorious five-in-a-row.",
                            "Securing a win is within reach for the <Player> with a potential five-in-a-row at <Move>."
                            "The <Player> can triumph with a well-executed five-in-a-row pattern starting at <Move>.",
                            "A decisive five-in-a-row at <Move> could seal the victory for the <Player>.",
                            "With a strategic placement at <Move>, the <Player> is poised to complete a five-in-a-row and win the game.",
                            "The opportunity for a game-winning five-in-a-row at <Move> is in the <Player>’s grasp.",
                            "The <Player> has a golden chance to win with a five-in-a-row sequence starting at <Move>.",
                            "Victory is a stone's throw away for the <Player> with a potential five-in-a-row at <Move>.",
                            "A win is on the horizon for the <Player> if they can form a five-in-a-row at <Move>.",
                            "The <Player> can clinch the game with a perfect five-in-a-row formation at <Move>.",
                            "At <Move>, the <Player> has the chance to secure victory with a five-in-a-row.",
                            "The <Player> is in a winning position with the potential for a five-in-a-row at <Move>.",
                            "Securing a five-in-a-row at <Move> would mean victory for the <Player>.",
                            "The <Player> is on the verge of winning with a potential five-in-a-row at <Move>.",
                            "A strategic move to <Move> could result in a five-in-a-row and a win for the <Player>.",
                            "The game is in the <Player>’s hands with a winning five-in-a-row at <Move>.",
                            "With the opportunity to create a five-in-a-row at <Move>, the <Player> is in a dominant position.",
                            "The <Player> can conclude the game with a victorious five-in-a-row at <Move>.",
                            "Victory is within the <Player>’s reach with a possible five-in-a-row starting at <Move>.",
                            "A five-in-a-row at <Move> would secure the win for the <Player>.",
                            "The <Player> has the chance to win the game with a five-in-a-row sequence at <Move>.",
                            ]
    
    anwser_empty = ["Currently, the <Player> lacks positions that could immediately result in open-three patterns or stronger formations.",
                    "At this moment, there are no specific spots on the board that offer the <Player> a direct path to creating open-three or more advanced patterns.",
                    "The board doesn’t present any advantageous positions for the <Player> to form open-threes or higher-order patterns right now.",
                    "For the <Player>, the board lacks clear opportunities to create open-three patterns or any stronger configurations.",
                    "No positions on the board stand out as immediate chances for the <Player> to achieve open-threes or more significant patterns.",
                    "The <Player> is currently without any special points that could lead directly to the formation of open-threes or more complex patterns.",
                    "At this stage, the board does not offer the <Player> any straightforward opportunities for creating open-threes or more advanced formations.",
                    "There are no distinct spots on the board that would allow the <Player> to quickly form open-threes or stronger patterns.",
                    "For the <Player>, the current board situation does not present any direct paths to achieving open-threes or more substantial formations.",
                    "The board is currently devoid of specific positions that would enable the <Player> to form open-threes or more complex patterns.",
                    ]
    
    def get_status_answer(poi_readable, player):
        status_answer_list = []
        if all([len(x)==0 for x in poi_readable.values()]):
            _ans = random.choice(anwser_empty)
            _ans = _ans.replace('<Player>', player)
            status_answer_list.append(_ans)
        else:
            status_answer_list.append(f'Some point of interest for the {player}:')
            for _pattern in ["Open-Three", "Open-Four"]:
                if len(poi_readable[_pattern]) > 0:
                    _ans = random.choice(answer_single_threat)
                    _ans = _ans.replace('<Player>', player).replace('<Pattern>', _pattern).replace('<Move>', ', '.join(poi_readable[_pattern]))
                    status_answer_list.append(_ans)
            for _pattern in ["Double-Open-Three", "Open-Four-Three", "Double-Open-Four"]:
                if len(poi_readable[_pattern]) > 0:
                    _ans = random.choice(answer_single_threat + answer_double_threat)
                    _ans = _ans.replace('<Player>', player).replace('<Pattern>', _pattern).replace('<Move>', ', '.join(poi_readable[_pattern]))
                    status_answer_list.append(_ans)
            for _pattern in ["Win-Five"]:
                if len(poi_readable[_pattern]) > 0:
                    _ans = random.choice(answer_five)
                    _ans = _ans.replace('<Player>', player).replace('<Pattern>', _pattern).replace('<Move>', ', '.join(poi_readable[_pattern]))
                    status_answer_list.append(_ans)

        return ' '.join(status_answer_list)
    
    _, poi_readable_player = compute_interest_points(np.array(board,dtype=np.int8), 1)
    _, poi_readable_opponent = compute_interest_points(np.array(board,dtype=np.int8), -1)
    inst = random.choice(instruction)
    ans_player = get_status_answer(poi_readable_player, 'current player')
    ans_opponent = get_status_answer(poi_readable_opponent, 'opponent')

    if random.random()>0.5:
        prompt = f"The current game board is <board>.\n{inst}"
        completion = f"{ans_player}\n{ans_opponent}"
    else:
        prompt = f"{inst}\nThe current game board is <board>."
        completion = f"{ans_player}\n{ans_opponent}"
    
    return {"canonicalBoard": board, "prompt": prompt, "completion": completion}

def worker(total_num, rank, lock):
    progress_bar = tqdm.tqdm(desc=f"Self-Play #{rank}", total=total_num, position=rank)
    progress_bar.set_lock(lock)
    data_count = 0
    with open(f"data/selfplay/selfplay_data_{rank}.json", 'r') as fin:
        with open(f"data/pretrain/pretrain_data_{rank}.json", 'w') as fout:
            for line in fin:
                record = json.loads(line)
                move_count = record["move_count"]
                for i in range(move_count):
                    _move_record = record[f"move{i+1}"]

                    # Suggest Move Data
                    counts = np.array(_move_record["counts"])
                    possible_moves = np.argwhere(counts > 0.05 * np.sum(counts)).reshape(-1).tolist()
                    for action in possible_moves:
                        pv = _move_record['pv'][action]
                        pv = pv if len(pv)>0 else None
                        data_suggest_move = template_suggest_move(_move_record["canonicalBoard"], action, pv)
                        fout.write(json.dumps(data_suggest_move) + '\n')
                        data_count += 1

                    # Status Analyze Data
                    data_status_analyze = template_describe_status(_move_record["canonicalBoard"])
                    fout.write(json.dumps(data_status_analyze) + '\n')
                    data_count += 1

                progress_bar.set_postfix(data_count=data_count)
                progress_bar.update(1)
    ...

if __name__=="__main__":
    mp.set_start_method('spawn')
    lock = mp.RLock()
    for i in range(12):
        p = mp.Process(target=worker, args=(1000, i, lock))
        p.start()
    for _ in range(12):
        p.join()
