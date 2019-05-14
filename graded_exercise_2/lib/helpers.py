import numpy as np
import os
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Latex

questions = list(map(str.strip, open(os.path.join('lib', 'theory_questions.txt'), 'r').readlines()))


def print_answers(answers):
    answers.extend(['X'] * (len(questions) - len(answers)))

    assert len(answers) == len(questions), "You do not have the correct amount of answers"
    def answer_to_symbol(answer):
        if answer.lower() in ('y', 'yes', 'true', 't'):
            return 'True'
        if answer.lower() in ('n', 'no', 'false', 'f'):
            return 'False'
        return 'Not answered'

    answer_markdown = "#### Your answers:\n"
    answer_markdown += "\n".join("{}. {}: **{}**".format(i + 1, question, answer)
            for i, (question, answer) in enumerate(zip(questions, map(answer_to_symbol, answers))))
    display(Markdown(answer_markdown))


def generate_answers_file(my_answers):
    answers_file = open('TF_answers', 'w')
    for ind, answer in enumerate(my_answers):
        answers_file.write(str(ind) + '\t' + answer + '\n')


def plot_grid_search_results(loss_array, lambdas, degrees):
    plt.figure(figsize=(8,10))
    plt.imshow(loss_array)
    plt.colorbar(orientation='horizontal')
    plt.xticks(np.arange(degrees.shape[0]), degrees, rotation=20)
    plt.yticks(np.arange(lambdas.shape[0]), lambdas, rotation=20)
    plt.xlabel('degree')
    plt.ylabel('lambda')
    plt.title('Validation loss for different lambda and degree')
    plt.show()


def show_theory_questions():
    question_markdown = "#### QUESTIONS\n"
    question_markdown += "\n".join("{}. {}".format(i + 1, question) for i, question in enumerate(questions))

    display(Markdown(question_markdown))

    return questions


def generate_answers_for_grading(scope):
    return
    # Now all this happens automatically
    sciper_number = scope.get('sciper_number')
    answers = scope.get('answers')

    file_path = 'answers_{}.npz'.format(sciper_number)
    np.savez(file_path, **answers)
    print("""
    Your answers have been saved to {0}.

    Please submit the following files ONLY:
        - graded_exercise_2.ipynb
        - {0}
    """.format(file_path))


def show_submission_instructions(scope):
    from .tests import get_submission_path
    submission_path = get_submission_path(scope)
    moodle_path = "https://moodle.epfl.ch/mod/assign/view.php?id=1023343"
    markdown = "#### You have reached the end of the tests.\n\n"
    markdown += "### [**Click here to go to moodle to upload your submission**]({}).\n\n".format(moodle_path)
    markdown += "Please submit the following files ONLY:\n"
    markdown += "- `graded_exercise_2.ipynb`\n"
    markdown += "- `{}` (automatically generated)\n".format(submission_path)
    display(Markdown(markdown))

