import openai
import json

def load_api_key(file_path='api_key.txt'):
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('api_key='):
                return line.strip().split('=', 1)[1]
    return None

openai.api_key = load_api_key()

if openai.api_key is None:
    print("Error: API key not found.")
    exit()

# activities contains all the activties in one string
def generate_activity_description(activity_name, num, activities, motion_description = None):
    if motion_description is not None:
        prompt = f'{num } prompts for {activity_name}. Motion description: {motion_description}'
    else:
        prompt = f'{num } prompts for {activity_name}' 
    
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
            {"role": "system", "content": "You are a prompt generator designed to generate textual description inputs for activtities. Do not provide anything other than prompt"},
            {"role": "user", "content": f"Prompts must be less than 15 words, contain description of the action with no description for envrionment. The prompt should only describe one person and an adjective describing how the person does the activity. Prompt cannot include more than one of the following activities: {activities}. Here are some example inputs: 'a person puts their hands together, leans forwards slightly, then swings the arms from right to left',  'a man is practicing the waltz with a partner', 'a man steps forward and does a handstand',   'a man rises from the ground, walks in a circle', 'a person jogs in place, slowly at first, then increases speed.',  'a man playing violin', 'a man playing guitar', 'a man doing skip rope', 'a man walks happily'."},
            {"role": "assistant", "content": "As an AI language model, I can certainly try to generate textual inputs based on the given activities"}, 
            # {"role": "user", "content": f'20  prompts for walking' },            
            # {"role": "assistant", "content": "A person takes slow, deliberate steps forward, placing one foot in front of the other.\nA man walks confidently with his head held high, his arms swinging freely at his sides.\nA person walks briskly, quickening their pace as they move forward.\nA woman takes small, careful steps, delicately navigating a rocky path.\nA man strolls leisurely, taking in the scenery around him as he walks.\nA person walks tentatively, testing the ground beneath their feet before taking each step.\nA woman walks with purpose, her strides long and powerful as she moves forward.\nA man shuffles his feet as he walks, his head down in deep thought.\nA person walks in a zigzag pattern, playfully avoiding obstacles in their path.\nA woman walks confidently in high heels, her steps graceful and fluid.\nA man walks in slow motion, his movements exaggerated and deliberate.\nA person takes long, purposeful strides as they walk, covering ground quickly.\nA woman walks cautiously, using a walking stick to steady herself.\nA man walks with a limp, favoring one leg as he moves forward.\nA person tiptoes quietly, trying not to make a sound as they walk.\nA woman walks with a bounce in her step, radiating energy and enthusiasm.\nA man walks with a cane, relying on the support of the sturdy wooden stick.\nA person walks with a limp, their gait uneven and unsteady.\nA woman walks along the beach, the sand soft and warm beneath her feet.\nA man walks briskly, his breath visible in the cold winter air.\n"},
            {"role": "user", "content": prompt }
        ]
    )
    # print(response)
    # description = response.choices[0].text.strip()
    return response