##INSTRUCTIONS##
You are tasked with identifying all the possible locations a user might perform a task in. You will be given an image and environment description and a task in JSON form with a name field and a brief description in the desc field. Identify potential locations in the image that the user may need to interact with to perform the task. Be as concise as possible, describing only the most important locations. Respond in JSON with these keys and values: "name": string, name of the location, "reason": string, why the user will interact with this location, "primitives": list[string], a list of all primitive motions or actions the user may need to do to perform the task at this location. This should be as exhaustive as possible while only listing general motions. These should all be motions or physical actions. For example, primitives could include "reaching arm up" or "sitting down". These should be as general of motions as possible while describing what the user might perform.

##EXAMPLES##
Input: A picture of a bathroom.
{
    "name": "Using the Toilet",
    "desc": "Using the toilet for personal needs"
}

Output:
[
    {
        "name": "toilet",
        "reason": "Conduct personal needs"
        "primitives": [
            "sit down",
            "stand up",
            "bend over"
        ],
        "name": "Sink",
        "primitives": [
            reach with arm,
            grasp,
        ]
    }
]

Input: A picture of a restaurant
{
    "location": "Dining",
    "desc": "Eating comfortably at the restaurant"
}


Output:
[
    {
        "location": "table",
        "reason": "Food will be served at the table"
        "primitives": [
            "sit down",
            "stand up",
            "grasp",
            "read in dark"
        ]
    }
]
