##INSTRUCTIONS##
You are tasked with identifying the potential tasks a user might perform in a given space. You will be given a set of images and a brief textual description of the environment and what the user intends to do. Identify all the potential tasks that might be performed within the environment depicted in the pictures given the provided description and items in the environment. Be as concise as possible. Describe only the most relevant tasks. Do not add any tasks that would be extraneous. Respond in JSON with these keys and values: "name": string, name of the task, "desc": string, brief description of what the task involves.

##EXAMPLES##
Input: an image of a bathroom
Output: [
    {
        "name": "Using the Toilet",
        "desc": "Using the toilet for personal needs"
    },
    {
        "name": "Washing Up",
        "desc": "Washing your face and body and freshening up in the morning"
    },
    {
        "name": "Taking care of Oral Hygiene",
        "desc": "Brushing teeth and rinsing your mouth"
    }
]

Input: An image of a restaurant I am going on a date at
Output: [
    {
        "name": "Dining",
        "desc": "Eating comfortably at the restaurant"
    },
    {
        "name": "Reading the Menu",
        "desc": "Checking the menu to know what to order"
    },
    {
        "name": "Chatting",
        "desc": "Talking with your date
    }
]
