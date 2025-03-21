from os.path import dirname, join
import time

from ..common.base_classes import SetupConfig
from ..common.constants.log_strings import CommonLogsStr
from ..common.utils.logging import get_glue_logger, set_logging_config
from ..common.utils.file import read_jsonl, yaml_to_class, yaml_to_dict, read_jsonl_row
from ..paramlogger import ParamLogger
from ..promptopt.constants import PromptOptimizationLiterals
from ..promptopt.techniques.common_logic import DatasetSpecificProcessing
from ..promptopt.utils import get_promptopt_class

PROMPT_POOL_DICT = {
    "final_prompt": "{instruction}\n{few_shot_examples}\n\n{answer_format}",
    "eval_prompt": "{instruction}\n\n[Question] {question}\n[Answer] ",
    "quest_reason_ans": "\n\n[Question] {question}\n[Answer] {answer}",
    "system_prompt": "You are a helpful assistant developed by OpenAI that can efficiently perform tasks as per instruction",
    "expert_profile": "You are a helpful assistant developed by OpenAI that can efficiently perform tasks as per instruction",
    "thinking_styles": [
        "How could I devise an experiment to help solve that problem?",
        "Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made.",
        "How could I measure progress on this problem?",
        "How can I simplify the problem so that it is easier to solve?",
        "What are the key assumptions underlying this problem?",
        "What are the potential risks and drawbacks of each solution?",
        "What are the alternative perspectives or viewpoints on this problem?",
        "What are the long-term implications of this problem and its solutions?",
        "How can I break down this problem into smaller, more manageable parts?",
        "Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying potential biases or flaws in thinking.",
        "Try creative thinking, generate innovative and out-of-the-box ideas to solve the problem. Explore unconventional solutions, thinking beyond traditional boundaries, and encouraging imagination and originality.",
        "Seek input and collaboration from others to solve the problem. Emphasize teamwork, open communication, and leveraging the diverse perspectives and expertise of a group to come up with effective solutions.",
        "Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements. Focuses on identifying the underlying causes, feedback loops, and interdependencies that influence the problem, and developing holistic solutions that address the system as a whole.",
        "Use Risk Analysis: Evaluate potential risks, uncertainties, and tradeoffs associated with different solutions or approaches to a problem. Emphasize assessing the potential consequences and likelihood of success or failure, and making informed decisions based on a balanced analysis of risks and benefits.",
        "Use Reflective Thinking: Step back from the problem, take the time for introspection and self-reflection. Examine personal biases, assumptions, and mental models that may influence problem-solving, and being open to learning from past experiences to improve future approaches.",
        "What is the core issue or problem that needs to be addressed?",
        "What are the underlying causes or factors contributing to the problem?",
        "Are there any potential solutions or strategies that have been tried before? If yes, what were the outcomes and lessons learned?",
        "What are the potential obstacles or challenges that might arise in solving this problem?",
        "Are there any relevant data or information that can provide insights into the problem? If yes, what data sources are available, and how can they be analyzed?",
        "Are there any stakeholders or individuals who are directly affected by the problem? What are their perspectives and needs?",
        "What resources (financial, human, technological, etc.) are needed to tackle the problem effectively?",
        "How can progress or success in solving the problem be measured or evaluated?",
        "What indicators or metrics can be used?",
        "Is the problem a technical or practical one that requires a specific expertise or skill set? Or is it more of a conceptual or theoretical problem?",
        "Does the problem involve a physical constraint, such as limited resources, infrastructure, or space?",
        "Is the problem related to human behavior, such as a social, cultural, or psychological issue?",
        "Does the problem involve decision-making or planning, where choices need to be made under uncertainty or with competing objectives?",
        "Is the problem an analytical one that requires data analysis, modeling, or optimization techniques?",
        "Is the problem a design challenge that requires creative solutions and innovation?",
        "Does the problem require addressing systemic or structural issues rather than just individual instances?",
        "Is the problem time-sensitive or urgent, requiring immediate attention and action?",
        "What kinds of solution typically are produced for this kind of problem specification?",
        "Given the problem specification and the current best solution, have a guess about other possible solutions.",
        "Let's imagine the current best solution is totally wrong, what other ways are there to think about the problem specification?",
        "What is the best way to modify this current best solution, given what you know about these kinds of problem specification?",
        "Ignoring the current best solution, create an entirely new solution to the problem.",
        "Let's think step by step.",
        "Let's make a step by step plan and implement it with good notion and explanation."
    ],
    "ans_delimiter_instruction": "",
    "meta_critique_template": "I'm trying to write a zero-shot instruction that will help the most capable and suitable agent to solve the task.\nMy current prompt is: \"{instruction}\"\nBut this prompt gets the following examples wrong: {examples}\nProvide detail feedback which identifies reasons where the instruction could have gone wrong.\nWrap each reason with <START> and <END>\n",
    "meta_positive_critique_template": "I'm trying to write a prompt for zero-shot instruction task that will help the most capable and suitable agent to solve the task.\nMy current prompt is:\n[CURRENT PROMPT] \"{instruction}\"\nNow this prompt got the following examples correct:\n[CORRECT EXAMPLES] {examples}\nSince you cant use these examples, analyse and understand characteristics/complexity and diversity of these examples and their reasoning chain and\naccordingly provide suggestions to further improve the prompt and make it better as a zero shot instruction task.\n  ",
    "critique_refine_template": "I'm trying to write a zero-shot instruction that will help the most capable and suitable agent to solve the task.\nMy current prompt is: \"{instruction}\"\nBut this prompt gets the following examples wrong: {examples}\nOn carefully analysing these examples, following are the critiques related to prompt {critique}\nUse the critique smartly, refine the current prompt to make sure we dont get these examples wrong.\nBased on the above information, Now I want you to write {steps_per_sample} different improved prompts.\nEach prompt should be wrapped with <START> and <END>.\n[Refined Prompts]:",
    "solve_template": "You are given a prompt instruction and the following {questions_batch_size} questions of the same task.\n[Instruction]: {instruction}\n\n[Question]: {questions}\n\n{answer_format}\n\n[Answers]:\n",
    "meta_sample_template": "You are given a task description and a prompt instruction and different styles known as meta prompts:\n[Task Description]: {task_description}\n[Meta Prompt]: {meta_prompts}\nNow you need to generate {num_variations} variations of following Instruction adaptively mixing meta prompt while keeping similar semantic meaning.\nMake sure to wrap each generated prompt with <START> and <END>\n[Prompt Instruction]: {prompt_instruction}\n[Generated Prompts]:",
    "intent_template": "You are given an instruction along description of task labelled as [Task Description]. For the given instruction, list out 3-5 keywords in comma separated format as [Intent] which define the characteristics or properties required by the about the most capable and suitable agent to solve the task using the instruction.\n\n\n[Task Description]: {task_description}\n[Instruction]: {instruction}\n\n\n[Intent]:",
    "expert_template": "For each instruction, write a high-quality description about the most capable and suitable agent to answer the instruction. In second person perspective.\\n\n\n[Instruction]: Make a list of 5 possible effects of deforestation.\\n\n[Agent Description]: You are an environmental scientist with a specialization in the study of ecosystems and their interactions with human activities. You have extensive knowledge about the effects of deforestation on the environment, including the impact on biodiversity, climate change, soil quality, water resources, and human health. Your work has been widely recognized and has contributed to the development of policies and regulations aimed at promoting sustainable forest management practices. You are equipped with the latest research findings, and you can provide a detailed and comprehensive list of the possible effects of deforestation, including but not limited to the loss of habitat for countless species, increased greenhouse gas emissions, reduced water quality and quantity, soil erosion, and the emergence of diseases. Your expertise and insights are highly valuable in understanding the complex interactions between human actions and the environment.\n\n\n[Instruction]: Identify a descriptive phrase for an eclipse.\\n\n[Agent Description]: You are an astronomer with a deep understanding of celestial events and phenomena. Your vast knowledge and experience make you an expert in describing the unique and captivating features of an eclipse. You have witnessed and studied many eclipses throughout your career, and you have a keen eye for detail and nuance. Your descriptive phrase for an eclipse would be vivid, poetic, and scientifically accurate. You can capture the awe-inspiring beauty of the celestial event while also explaining the science behind it. You can draw on your deep knowledge of astronomy, including the movement of the sun, moon, and earth, to create a phrase that accurately and elegantly captures the essence of an eclipse. Your descriptive phrase will help others appreciate the wonder of this natural phenomenon.\n\n\n\n[Instruction]: Identify the parts of speech in this sentence: \\\"The dog barked at the postman\\\".\\n\n[Agent Description]: You are a linguist, well-versed in the study of language and its structures. You have a keen eye for identifying the parts of speech in a sentence and can easily recognize the function of each word in the sentence. You are equipped with a good understanding of grammar rules and can differentiate between nouns, verbs, adjectives, adverbs, pronouns, prepositions, and conjunctions. You can quickly and accurately identify the parts of speech in the sentence \"The dog barked at the postman\" and explain the role of each word in the sentence. Your expertise in language and grammar is highly valuable in analyzing and understanding the nuances of communication.\n\n\n[Instruction]: {task_description}\n[Agent Description]:",
    
    "examples_critique_template": "You are an expert example selector who can help in selection of right in-context examples to help the most suitable agent solve this problem.\nYou are also given the prompt instruction which is used to solve this task\n[Prompt]: {prompt}\nYou are given the task description of the task:\n[Task Description]: {task_description}\nI'm trying to write a few shots prompt using {num_examples} in-context examples to effectively solve any questions of the above task.\nMy current {num_examples} in-context examples set are: {examples}\nThink of analysing, understanding and creating examples of task on the criteria of diversity of types of examples, complexity of the nature/characteristics of the examples and relevance/compatibility to the whole example set in total.\nOutput all the suggestions/ improvement which could be made to improve each individual example of the whole example selection set.",
    
    "examples_critique_template_zero_shot": "You are an expert example selector who can help in selection of right in-context examples to help the most suitable agent solve this problem.\nYou are also given the prompt instruction which is used to solve this task\n[Prompt]: {prompt}\nYou are given the task description of the task:\n[Task Description]: {task_description}\nI'm trying to write a few shots prompt using {num_examples} in-context examples to effectively solve any questions of the above task.\nThink of analysing, understanding and creating examples of task on the criteria of diversity of types of examples, complexity of the nature/characteristics of the examples and relevance/compatibility to the whole example set in total.\nOutput all the suggestions/ improvement which could be made to improve each individual example of the whole example selection set.",
    
    "examples_optimization_template": "You are an expert example selector who can help in selection of right in-context examples to help the agent solve this problem.\nYou are also given the prompt instruction which is used to solve this task\n[Prompt]: {prompt}\nYou are given the description of the task:\n[Task Description]: {task_description}\nI'm trying to write a few shots prompt using {num_examples} in-context examples to effectively solve any questions of the above task.\nMy current {num_examples} in-context examples set are: {examples}\nYou are also given a set of suggestions/improvements which could be made to improve each individual example of the whole example selection set:\n[SUGGESTION/IMPROVEMENT]: {critique}\nBased on the above information, use all of it smartly and diligently to carefully create new set of {num_examples}, which follow these suggestion and improvements.\nMake sure to output each example wrapped with <START> and <END>.\n\nNew examples should follow this format strictly:\n\n[Question] followed by question part of the example\n[Answer] followed by the all the steps of logic reasoning statements related to answer. The final answer as \"<ANS_START>[answer]<ANS_END>\"\n\nFor Example: <START>\n{gt_example}\n<END>\n\n[New Examples]:",
    
    "generate_reason_template": "You are given a task description and instruction followed by a set of correct examples of the task.\n\n[Task Description]: {task_description}\n\n[Instruction]: {instruction}\n\nEach example has a question denoted by question [Question] and a final answer [Answer] .\n\n[Question]: {question}\n\n[Answer]: {answer}\n\nNow your task is to generate a reasoning chain that contains the steps, logical pathway followed to arrive at the correct answer, assuming the necessary domain knowledge is present as part of the question and task description.\n\nMake sure it is specific, non-ambiguous, complete, and specifies all the logic and steps required to reach the final answer.\n\n[Improved Reasoning Chain]:",
    
    "reason_optimization_template": "You are given a task description and instructions of given task\n\n[Task Description]: {task_description}\n\n[Instruction]: {instruction}\n\nEach example has a question denoted by a question [Question] and a final answer [Answer].\n\n[Question]: {question}\n\n[Answer]: {answer}\n\nPlease explain your reasoning behind reaching the answer given in a concise, complete, and coherent text of reasoning that contains all the steps or logical pathways followed. Ensure it is specific and non-ambiguous, and assume the necessary domain knowledge is in the question and task description.\n\n[Improved Reasoning Chain]:"
}

class GluePromptOpt:
    """
    This class is trigger point for any prompt optimization method. Different prompt optimization techniques are
    represented by different classes. This class collates all the user configs present in different yaml files and
    other boilerplate code. Any of supported prompt optimization techniques can be triggered by this class.
    """
    BEST_PROMPT = None
    EXPERT_PROFILE = None
    data_processor = None
    iolog = ParamLogger()

    class EvalLiterals:
        IS_CORRECT = "is_correct"
        PREDICTED_ANS = "predicted_ans"
        LLM_OUTPUT = "llm_output"

    def __init__(self,
                 prompt_config_path: str,
                 setup_config_path: str,
                 dataset_jsonl: str,
                 data_processor: DatasetSpecificProcessing,
                 dataset_processor_pkl_path: str = None,
                 prompt_pool_path: str = None):
        """
        Collates all the configs present in different yaml files. Initialize logger, de-serialize pickle file that has
        class/method for dataset processing (for given dataset).

        :param llm_config_path: Path to yaml file that has LLM related configs.
        :param prompt_config_path: Path to yaml file that has prompt templates for the given techniques.
        :param setup_config_path: Path to yaml file that has user preferences.
        :param dataset_jsonl: Path to jsonl file that has dataset present in jsonl format.
        :param data_processor: object of DatasetSpecificProcessing class, which has data handling methods which are
        specific to that dataset
        :param dataset_processor_pkl_path: Path to pickle file that has object of class DatasetSpecificProcessing
                                           serialized.
        :param prompt_pool_path: Path to yaml file that has prompts
        """
        if dataset_jsonl != None:
            if data_processor:
                self.data_processor = data_processor
            else:
                raise ValueError("data_processor is None. Please provide data_processor object")
                # with open(dataset_processor_pkl_path, "rb") as file:
                #     self.data_processor = pickle.load(file)  # datatype: class DatasetSpecificProcessing

        prompt_config_dict = yaml_to_dict(prompt_config_path)
        prompt_opt_cls, prompt_opt_hyperparam_cls, promptpool_cls = get_promptopt_class(
            prompt_config_dict[PromptOptimizationLiterals.PROMPT_TECHNIQUE_NAME])

        self.setup_config = yaml_to_class(setup_config_path, SetupConfig)
        self.prompt_opt_param = yaml_to_class(prompt_config_path, prompt_opt_hyperparam_cls)
        # current_dir = dirname(__file__)
        # default_yaml_path = join(current_dir,
        #                          "techniques",
        #                          prompt_config_dict[PromptOptimizationLiterals.PROMPT_TECHNIQUE_NAME],
        #                          "prompt_pool.yaml")

        self.prompt_pool = promptpool_cls(**PROMPT_POOL_DICT) #yaml_to_class(prompt_pool_path, promptpool_cls, default_yaml_path)

        if dataset_jsonl != None:
            dataset = read_jsonl(dataset_jsonl)
        self.prompt_opt_param.answer_format += self.prompt_pool.ans_delimiter_instruction
        base_path = join(self.setup_config.dir_info.base_dir, self.setup_config.experiment_name)
        set_logging_config(join(base_path, self.setup_config.dir_info.log_dir_name),
                           self.setup_config.mode)
        self.logger = get_glue_logger(__name__)

        if dataset_jsonl != None:
            if len(dataset) < self.prompt_opt_param.seen_set_size:
                self.prompt_opt_param.seen_set_size = len(dataset)
                self.logger.info(f"Dataset has {len(dataset)} samples. However values for seen_set_size is "
                                f"{self.prompt_opt_param.seen_set_size}. Hence resetting seen_set_size"
                                f" to {len(dataset)}")

        if self.prompt_opt_param.few_shot_count > self.prompt_opt_param.seen_set_size:
            self.prompt_opt_param.few_shot_count = self.prompt_opt_param.seen_set_size
            self.logger.info(f"Value set for few_shot_count is {self.prompt_opt_param.few_shot_count}. "
                             f"However values for seen_set_size is {self.prompt_opt_param.seen_set_size}. "
                             f"Hence resetting few_shot_count to {self.prompt_opt_param.few_shot_count}")

        if dataset_jsonl != None:
            training_dataset = dataset[:self.prompt_opt_param.seen_set_size]
        else:
            training_dataset = None
        self.logger.info(f"Setup configurations parameters: {self.setup_config} \n{CommonLogsStr.LOG_SEPERATOR}")
        self.logger.info(f"Prompt Optimization parameters: {self.prompt_opt_param} \n{CommonLogsStr.LOG_SEPERATOR}")

        # This iolog is going to be used when doing complete evaluation over test-dataset
        self.iolog.reset_eval_glue(join(base_path, "evaluation"))

        self.prompt_opt = prompt_opt_cls(training_dataset, base_path, self.setup_config,
                                         self.prompt_pool, self.data_processor, self.logger)

    def get_best_prompt(self,use_examples=False,run_without_train_examples=False,generate_synthetic_examples=False):# -> (str, Any):
        """
        Call get_best_prompt() method of class PromptOptimizer & return its value.
        :return: (best_prompt, expert_profile)
            best_prompt-> Best prompt for a given task description
            expert_profile-> Description of an expert who is apt to solve the task at hand. LLM would be asked to take
            identity of described in expert_profile.
        """
        start_time = time.time()
        self.BEST_PROMPT, self.EXPERT_PROFILE = self.prompt_opt.get_best_prompt(self.prompt_opt_param,use_examples=use_examples,run_without_train_examples=run_without_train_examples,generate_synthetic_examples=generate_synthetic_examples)

        self.logger.info(f"Time taken to find best prompt: {(time.time() - start_time)} sec")
        return self.BEST_PROMPT, self.EXPERT_PROFILE

    def evaluate(self, test_dataset_jsonl: str) -> float:
        """
        Evaluate the performance of self.BEST_PROMPT over test dataset. Return the accuracy.

        :param test_dataset_jsonl: Path to jsonl file that has test dataset
        :return: Percentage accuracy
        """

        start_time = time.time()
        self.logger.info(f"Evaluation started {CommonLogsStr.LOG_SEPERATOR}")
        if not self.BEST_PROMPT:
            self.logger.error("BEST_PROMPT attribute is not set. Please set self.BEST_PROMPT attribute of this object, "
                              "either manually or by calling get_best_prompt() method.")
            return

        total_correct = 0
        total_count = 0
        for json_obj in read_jsonl_row(test_dataset_jsonl):
            answer = self.predict_and_access(json_obj[DatasetSpecificProcessing.QUESTION_LITERAL],
                                             json_obj[DatasetSpecificProcessing.FINAL_ANSWER_LITERAL])
      
            total_correct += answer[self.EvalLiterals.IS_CORRECT]
            total_count += 1
            result = {"accuracy": f"{total_correct}/{total_count} : {total_correct/total_count}%",
                      "predicted": answer[self.EvalLiterals.PREDICTED_ANS],
                      "actual": json_obj[DatasetSpecificProcessing.FINAL_ANSWER_LITERAL]}
            self.iolog.append_dict_to_chained_logs(result)
            self.logger.info(result)

        self.iolog.dump_chained_log_to_file(file_name=f"eval_result_{self.setup_config.experiment_name}")
        self.logger.info(f"Time taken for evaluation: {(time.time() - start_time)} sec")
        return total_correct / total_count

    @iolog.log_io_params
    def predict_and_access(self, question: str, gt_answer: str) -> (bool, str, str):
        """
        For the given input question, get answer to it from LLM, using the BEST_PROMPT & EXPERT_PROFILE
        computes earlier.

        :param question: Question to be asked to LLM, to solve
        :param gt_answer: Ground truth, final answer.
        :return:  (is_correct, predicted_ans, llm_output)
                is_correct -> Tells if prediction by LLM was correct.
                predicted_ans -> is the actual predicted answer by LLM.
                llm_output -> Output text generated by LLM for the given question
        :rtype: (bool, str, str)
        """
        final_prompt = self.prompt_pool.eval_prompt.format(instruction=self.BEST_PROMPT,
                                                           question=question)
        llm_output = self.prompt_opt.chat_completion(user_prompt=final_prompt, system_prompt=self.EXPERT_PROFILE)
        
        is_correct, predicted_ans = self.data_processor.access_answer(llm_output, gt_answer)
        return {self.EvalLiterals.IS_CORRECT: is_correct,
                self.EvalLiterals.PREDICTED_ANS: predicted_ans,
                self.EvalLiterals.LLM_OUTPUT: llm_output}

