import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import textwrap
from scipy import stats
from scipy.stats import binomtest
from statsmodels.stats.contingency_tables import mcnemar


def display_images(question):
    path_to_cropped = os.path.join('./images/', question, 'cropped.png')
    path_to_original = os.path.join('./images/', question, 'original.jpeg')

    image1 = Image.open(path_to_cropped)
    image2 = Image.open(path_to_original)

    fig, axes = plt.subplots(1, 2, figsize=(6, 4))

    # Plot cropped img
    axes[0].imshow(image1)
    axes[0].axis('off')  # Remove axis ticks and labels
    axes[0].set_title('Cropped')

    # Plot original img
    axes[1].imshow(image2)
    axes[1].axis('off')
    axes[1].set_title('Original')
    
    plt.tight_layout()
    plt.show()
    
def preprocess_df(df, question):
    q = df.filter(regex=f'^{question}', axis=1) # get all subquestions of Q1
    q = q.drop(index=1) # remove second row that question ID
    q = q.dropna() # remove NaN rows
    q.reset_index(drop=True, inplace=True) # reset index values
    
    return q

def plot_answer_distributions(df, question_parts=['.1', '.2', '.3', '.4', '.5'], save_fig=False):
    sns.set_style('whitegrid')
    plt.rcParams['font.family'] = 'serif'
    
    # Define mappings and orders directly inside the function
    mappings_and_orders = {
        '.1': {
            'mapping': {'A real car': 'Real', 'A toy car': 'Toy', 'I am unsure': 'Unsure'},
            'order': ['Real', 'Toy', 'Unsure']
        },
        '.2': {
            'mapping': {'Object': 'Object', 'Surrounding (Environment around the car)': 'Context',
                        'Both equally': 'Both', 'If you selected unsure in previous question, choose this option': 'Unsure'},
            'order': ['Object', 'Context', 'Both', 'Unsure']
        },
        '.3': {
            'mapping': {'A real car': 'Real', 'A toy car': 'Toy', 'I am still unsure': 'Unsure'},
            'order': ['Real', 'Toy', 'Unsure']
        },
        '.4': {
            'mapping': {'Object': 'Object', 'Surrounding (Environment around the car)': 'Context',
                        'Both equally': 'Both', 'If you selected unsure in previous question, choose this option': 'Unsure'},
            'order': ['Object', 'Context', 'Both', 'Unsure']
        },
        '.5': {
            'mapping': {'Context/environment in which the (toy) car can be seen': 'Context',
                        'Size in relation to other objects': 'Size', 'Colour': 'Colour',
                        'Details on the (toy) car': 'Object details', 'Blurriness or sharpness of fore/background': 'Image clarity',
                        'Other': 'Other', 'Texture/reflections or graininess': 'Texture'},
            'order': ['Context', 'Size', 'Colour', 'Object details', 'Image clarity', 'Other', 'Texture']
        }
    }

    # Calculate global max count across all mappings and orders
    global_max_count = 0
    for question_suffix, info in mappings_and_orders.items():
        for col in df.columns:
            if col.endswith(question_suffix):
                answers = df[col].dropna()
                if question_suffix == '.5':  # Handle multiple selections for .5 questions
                    answers = answers.str.split(',').explode()
                mapped_answers = answers.map(lambda x: info['mapping'].get(x, x)).dropna()
                max_count = mapped_answers.value_counts().max()
                if max_count > global_max_count:
                    global_max_count = max_count

    num_questions = len(question_parts)
    fig, axes = plt.subplots(1, num_questions, figsize=(5 * num_questions, 6) if num_questions > 1 else (8, 6), sharey=True)
    if num_questions == 1:
        axes = [axes]  # Ensure axes is always iterable

    # Plotting for specified question parts
    for idx, part in enumerate(question_parts):
        filtered_columns = [col for col in df.columns if col.endswith(part)]
        for question_col in filtered_columns:
            mapping_info = mappings_and_orders.get(part, {'mapping': {}, 'order': []})
            question_text = df.loc[0, question_col]
            answers = df.loc[1:, question_col]  # Assuming answers start from the second row
            if part == '.5':
                answers = answers.str.split(',').explode()
            mapped_answers = answers.map(lambda x: mapping_info['mapping'].get(x, x)).dropna()

            ax = axes[idx]
            sns.countplot(x=mapped_answers, ax=ax, order=mapping_info['order'])
            title = f'{question_col}: {question_text}'
            wrapped_title = textwrap.fill(title, width=50)
            ax.set_title(wrapped_title, fontsize=12)
            ax.set_ylim(0, global_max_count + global_max_count * 0.1)  # Add a little extra space
            ax.set_xlabel('Answer', fontsize=20)
            ax.set_ylabel('Count', fontsize=20)
            ax.tick_params(axis='x', rotation=45, labelsize=16)

            # Annotate counts above bars
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(f'Q{question_col[1]}{"_".join(question_parts)}.png')
    
    plt.show()