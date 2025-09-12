import os
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
import gradio as gr

from src.evaluator import EnvironmentEvaluator
from src.user_modeler import UserModeler

load_dotenv()
API_KEY = os.environ.get("OPENAI_API_KEY")

# global so we can continuously access alter
global MODELER
global EVALUATOR

# DEFAULT_LOAD_DIR = Path(r"temp/formative/example/baseline/house")
# DEFAULT_SAVE_DIR = Path("temp/formative/out")
# DEFAULT_USER_MODEL_JSON_PATH = Path("temp/formative/example/user_capabilities.json")

# DEFAULT_LOAD_DIR = Path("hidden_data/formative/experiment/testing_baselines")
# DEFAULT_SAVE_DIR = Path("hidden_data/formative/experiment/testing_results")
# DEFAULT_USER_MODEL_JSON_PATH = Path("hidden_data/formative/experiment/user_model.json")

DEFAULT_LOAD_DIR = Path("hidden_data/final/P11/training_baselines")
DEFAULT_SAVE_DIR = Path("hidden_data/final/P11/training_results")
DEFAULT_USER_MODEL_JSON_PATH = Path("hidden_data/final/P11/user_model.json")

DEFAULT_SAVE_DIR.mkdir(exist_ok=True)

def gen_annotatedimage_segs():
    global EVALUATOR
    env_concerns = EVALUATOR.get_env_concerns()
    env = EVALUATOR.get_env()
    env_img = env.get_env_imgs()[0]
    img = env_img.img

    sections = []
    for env_concern in env_concerns:
        if env_concern.img_name != env_img.name:
            continue
        if not env_concern.is_concern:
            continue

        mask = np.zeros(img.size)
        for mark in env_concern.locs:
            mask += env_img.masks[mark - 1].T * 0.6
        name = env_concern.name.rstrip()

        # approved indicator
        if not env_concern.approved:
            name += " ☐"
        else:
            name += " ☑"

        sections.append((mask.T, name))

    return img, sections


def gen_user_model_text() -> str:
    global MODELER
    user_model = MODELER.get_user_model()

    final_str = ""
    for att in user_model:
        final_str += f"**{att.name}** ({att.affected_part}) - {att.desc}\n- Frequent: {att.frequent}\n\n"

    return final_str


def gen_concern_text(concern) -> str:
    global EVALUATOR
    env_tasks = EVALUATOR.get_env_tasks()

    tasks = [env_tasks[idx] for idx in concern.task_idxs]

    final_str = f"# **{concern.name}**\n{concern.desc}\n\n## Affected Tasks:\n"

    for task in tasks:
        final_str += f"- {task.name} - {task.desc}\n"
        for primitive in task.primitives:
            final_str += f"  - {primitive}\n"

    return final_str


def select_user_model_type(init_user_model):
    return gr.File(visible=not init_user_model), gr.Textbox(visible=init_user_model)


def init_evaluation(
    eval_memory_path,
    user_model_json_path,
    user_model_chat_history,
):
    gr.Info("Initializing system")
    global MODELER
    global EVALUATOR

    EVALUATOR = EnvironmentEvaluator(API_KEY, memory_dir=eval_memory_path)
    MODELER = UserModeler(API_KEY, memory_path=user_model_json_path)
    EVALUATOR.update_user_model(MODELER.get_user_model(as_dict=True))

    seg = gen_annotatedimage_segs()

    user_model_chat_history.append(
        {"role": "assistant", "content": gen_user_model_text()}
    )

    return (
        gr.Column(visible=False),
        gr.Column(visible=True),
        seg,
        user_model_chat_history,
    )


def select_section(evt: gr.SelectData, anns):
    global MODELER
    global EVALUATOR
    env_concerns = EVALUATOR.get_env_concerns()

    # TODO: might have to be changed
    concern_name = anns[1][evt.index][1]
    concern_name = " ".join(concern_name.split(" ")[:-1])  # cuts out the little icons
    for idx, concern in enumerate(env_concerns):
        if concern.name == concern_name:
            break

    # instantly mark as approved
    EVALUATOR.approve_concern(idx, concern.is_concern, concern.user_rating)
    seg = gen_annotatedimage_segs()
    return (
        seg,
        gen_concern_text(concern),
        concern.is_concern,
        concern.user_rating,
    )


def update_user_model(message, chat_history):
    if message == "":
        return "", chat_history

    chat_history.append({"role": "user", "content": message})

    global MODELER
    global EVALUATOR

    MODELER.update_user_model(message)
    # EVALUATOR.update_user_model(
    #     MODELER.get_user_model(as_dict=True)
    # )  # update the concerns too
    chat_history.append({"role": "assistant", "content": gen_user_model_text()})

    return "", chat_history

def add_general_env_feedback(message, chat_history, user_model_chat_history):
    if message == "":
        return "", chat_history

    chat_history.append({"role": "user", "content": message})

    global MODELER
    global EVALUATOR

    MODELER.update_by_env_feedback(EVALUATOR.get_env(), message)

    user_model_chat_history.append(
        {"role": "assistant", "content": gen_user_model_text()}
    )

    return "", chat_history, user_model_chat_history


def approve_concern(concern_str, is_concern, user_rating):
    if len(concern_str) == 0:
        # if nothing was selected
        print("No concern selected.")
        seg = gen_annotatedimage_segs()
        return seg

    global EVALUATOR

    # strips out all the markdown formatting
    curr_concern_name = (
        concern_str.split("\n")[0].replace("**", "").replace("#", "").strip().lower()
    )
    env_concerns = EVALUATOR.get_env_concerns()

    concern_idx = -1
    for i, concern in enumerate(env_concerns):
        if concern.name.lower() == curr_concern_name:
            concern_idx = i
            break

    if concern_idx == -1:
        print(f"Concern does not exist to approve: {curr_concern_name}")
        seg = gen_annotatedimage_segs()
        return seg

    EVALUATOR.approve_concern(concern_idx, is_concern, user_rating)

    seg = gen_annotatedimage_segs()

    return seg


def add_concern(msg, chat_history):
    global EVALUATOR

    img_name = (
        EVALUATOR.get_env().get_env_imgs()[0].name
    )  # since we're only evaluating 1 image at a time
    added_concern = EVALUATOR.add_concern(img_name, msg)

    chat_history.append({"role": "user", "content": msg})
    chat_history.append(
        {
            "role": "assistant",
            "content": f"**{added_concern.name}** - {added_concern.desc}",
        }
    )

    seg = gen_annotatedimage_segs()

    return "", chat_history, seg


def update_user_model_by_env_concerns(
    in_path: str, save_dir: str, user_model_json_path: str
):
    gr.Info("Began user model saving.")
    global EVALUATOR
    global MODELER

    env = EVALUATOR.get_env()
    env_concerns = EVALUATOR.get_env_concerns()
    MODELER.update_by_env_ann(env, env_concerns)

    in_path = Path(in_path)
    save_dir = Path(save_dir)
    user_model_json_path = Path(user_model_json_path)

    save_path = save_dir / ("formative_" + in_path.stem)

    EVALUATOR.save_memory(save_path)

    MODELER.save_memory(user_model_json_path)
    gr.Info(f"User model saved at {str(user_model_json_path)}. \nEvaluator information saved at {str(save_dir)}")


with gr.Blocks(fill_width=True) as demo:
    with gr.Column(visible=True) as upload_col:
        evaluator_memory_path = gr.Textbox(
            label="Evaluator Memory Directory Path",
            value=str(DEFAULT_LOAD_DIR.absolute()) + "/",
        )
        evaluator_save_dir = gr.Textbox(
            label="Save Directory", value=str(DEFAULT_SAVE_DIR.absolute()) + "/"
        )
        user_model_json_path = gr.Textbox(
            label="User Model JSON Path",
            value=str(DEFAULT_USER_MODEL_JSON_PATH.absolute()),
        )

        evaluate_env_button = gr.Button("Evaluate Environment")

    with gr.Column(visible=False) as evaluate_col:
        with gr.Row() as org_row:
            with gr.Column(scale=1) as user_input_col:
                with gr.Tab("View Concern"):
                    with gr.Accordion("Environmental Concern Information", open=True):
                        # label doesnt work like other components so in accordion
                        env_concern_textbox = gr.Markdown(
                            value=" ", max_height=400, line_breaks=True
                        )
                    with gr.Column() as is_concern_col:
                        with gr.Row() as approve_row:
                            is_concern = gr.Radio(label="Concern Validity", choices=[("Is Concern", True), ("Is Not Concern", False)], value=True)
                        concern_rating = gr.Textbox(
                            label="Environmental Concern Feedback", scale=4
                        )
                        submit_concern_feedback = gr.Button(
                            value="Submit Concern Feedback", scale=1
                        )

                with gr.Tab("New Concern"):
                    add_concern_chatbot = gr.Chatbot(
                        type="messages", label="General Concerns"
                    )
                    add_concern_msg = gr.Textbox(label="New Concern Info", scale=4)
                    submit_new_concern = gr.Button(value="Submit New Concern", scale=1)

                with gr.Tab("General Feedback"):
                    gen_env_feedback_chatbot = gr.Chatbot(type="messages", label="General Comments")
                    gen_env_feedback_msg = gr.Textbox(label="Environment Feedback")
                    submit_gen_env_feedback = gr.Button(value="Submit General Environment Feedback", scale=1)

                with gr.Accordion("User Model", open=True):
                    user_model_chatbot = gr.Chatbot(type="messages", label="User Model")
                    user_model_msg = gr.Textbox(label="User Model Feedback")
                    user_model_msg.submit(
                        update_user_model,
                        inputs=[user_model_msg, user_model_chatbot],
                        outputs=[user_model_msg, user_model_chatbot],
                    )
                    submit_user_model_feedback = gr.Button(value="Submit User Model Feedback", scale=1)
                    submit_user_model_feedback.click(
                        update_user_model,
                        inputs=[user_model_msg, user_model_chatbot],
                        outputs=[user_model_msg, user_model_chatbot],
                    )

            with gr.Column(scale=2) as ann_col:
                ann_image = gr.AnnotatedImage(height=700)
                update_by_env_concerns = gr.Button(value="Save and Update User Model")

        update_by_env_concerns.click(
            update_user_model_by_env_concerns,
            inputs=[evaluator_memory_path, evaluator_save_dir, user_model_json_path],
        )

        concern_rating.submit(
            approve_concern,
            inputs=[env_concern_textbox, is_concern, concern_rating],
            outputs=[ann_image],
        )

        is_concern.change(
            approve_concern,
            inputs=[env_concern_textbox, is_concern, concern_rating],
            outputs=[ann_image],
        )

        submit_concern_feedback.click(
            approve_concern,
            inputs=[env_concern_textbox, is_concern, concern_rating],
            outputs=[ann_image],
        )

        ann_image.select(
            select_section,
            inputs=[ann_image],
            outputs=[ann_image, env_concern_textbox, is_concern, concern_rating],
        )

        add_concern_msg.submit(
            add_concern,
            inputs=[add_concern_msg, add_concern_chatbot],
            outputs=[add_concern_msg, add_concern_chatbot, ann_image],
        )

        submit_new_concern.click(
            add_concern,
            inputs=[add_concern_msg, add_concern_chatbot],
            outputs=[add_concern_msg, add_concern_chatbot, ann_image],
        )

        gen_env_feedback_msg.submit(
            add_general_env_feedback,
            inputs=[gen_env_feedback_msg, gen_env_feedback_chatbot, user_model_chatbot],
            outputs=[gen_env_feedback_msg, gen_env_feedback_chatbot, user_model_chatbot]
        )

        submit_gen_env_feedback.click(
            add_general_env_feedback,
            inputs=[gen_env_feedback_msg, gen_env_feedback_chatbot, user_model_chatbot],
            outputs=[gen_env_feedback_msg, gen_env_feedback_chatbot, user_model_chatbot]
        )

    evaluate_env_button.click(
        init_evaluation,
        inputs=[evaluator_memory_path, user_model_json_path, user_model_chatbot],
        outputs=[
            upload_col,
            evaluate_col,
            ann_image,
            user_model_chatbot,
        ],
    )

if __name__ == "__main__":
    demo.launch()
