MAX_NUM_SEQS = 8
MAX_MODEL_LEN = 16384
pre = 1.5

part = 1

is_submission = False
llm_model_pth = "/kaggle/input/qwen-3/transformers/30b-a3b-fp8/1"
is_moe = "a" in llm_model_pth.split("transformers/")[1]
from vllm import LLM, SamplingParams
import os
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
import gc
import time
import warnings

import pandas as pd
import polars as pl
import numpy as np

import torch
import kaggle_evaluation.aimo_2_inference_server

pd.set_option('display.max_colwidth', None)
start_time = time.time()
cutoff_time = start_time + (4 * 60 + 45) * 60
cutoff_times = [int(x) for x in np.linspace(cutoff_time, start_time + 180 * 60, 50 + 1)]

warnings.simplefilter('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


llm = LLM(
    llm_model_pth,
    # dtype="half",                # The data type for the model weights and activations
    max_num_seqs=MAX_NUM_SEQS,   # Maximum number of sequences per iteration. Default is 256
    max_model_len=MAX_MODEL_LEN, # Model context length
    trust_remote_code=True,      # Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer
    tensor_parallel_size=4,      # The number of GPUs to use for distributed execution with tensor parallelism
    enable_expert_parallel = is_moe,
    gpu_memory_utilization=0.90, # The ratio (between 0 and 1) of GPU memory to reserve for the model
    seed=2024,
)
tokenizer = llm.get_tokenizer()
import re
import keyword


def extract_boxed_text(text):
    pattern = r'oxed{(.*?)}'
    matches = re.findall(pattern, text)
    if not matches:
        return ""
    for match in matches[::-1]:
        if match != "":
            return match
    return ""


from collections import Counter
import random
def select_answer(answers):
    counter = Counter()
    for answer in answers:
        try:
            if int(answer) == float(answer):
                if answer != 0:
                    counter[int(answer)] += 1 + random.random() / 1_000
        except:
            pass
    if not counter:
        return 210
    _, answer = sorted([(v,k) for k,v in counter.items()], reverse=True)[0]
    return answer%1000
def batch_message_generate(list_of_messages) -> list[list[dict]]:
    max_tokens = MAX_MODEL_LEN
    if time.time() > cutoff_times[-1]:
        print("Speedrun")
        max_tokens = 2 * MAX_MODEL_LEN // 3

    sampling_params = SamplingParams(
        temperature=1.0,
        skip_special_tokens=True,
        presence_penalty = pre,
        max_tokens=max_tokens,
    )   
    
    list_of_texts = [
        tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking = True
        )
        for messages in list_of_messages
    ]

    request_output = llm.generate(
        prompts=list_of_texts,
        sampling_params=sampling_params,
    )
    
    print([len(single_request_output.outputs[0].token_ids) for single_request_output in request_output])

    sort_keys_and_list_of_messages = []

    for messages, single_request_output in zip(list_of_messages, request_output):
        # print()
        # print(single_request_output.outputs[0].text)
        # print()
        messages.append({'role': 'assistant', 'content': single_request_output.outputs[0].text})

        sort_keys_and_list_of_messages.append(
            (
                len(single_request_output.outputs[0].token_ids),
                messages
            )
        )

    print([sort_key for sort_key, _ in sort_keys_and_list_of_messages])
    sort_keys_and_list_of_messages.sort(key=lambda sort_key_and_messages: sort_key_and_messages[0])
    print([sort_key for sort_key, _ in sort_keys_and_list_of_messages])

    list_of_messages = [messages for _, messages in sort_keys_and_list_of_messages]
    
    return list_of_messages
def batch_message_filter(list_of_messages) -> tuple[list[list[dict]], list[str]]:
    extracted_answers = []
    list_of_messages_to_keep = []
    for messages in list_of_messages:
        answer = extract_boxed_text(messages[-1]['content'])
        if answer:
            extracted_answers.append(answer)
        else:
            list_of_messages_to_keep.append(messages)
    return list_of_messages_to_keep, extracted_answers
def create_starter_messages(question, index):
    return(
        [
            {"role": "user", "content": question}
        ]
    )
def predict_for_question(question: str) -> int:
    import os

    if time.time() > cutoff_time:
        return 210
    
    print(question)

    if is_submission and not os.getenv('KAGGLE_IS_COMPETITION_RERUN') and "Triangle" not in question and "airline" not in question:
        return 210
    success = llm.llm_engine.reset_prefix_cache()
    if not success:
        print("failed to reset prefix cache")
    num_seqs = MAX_NUM_SEQS
    if time.time() > cutoff_times[-1]:
        num_seqs = 2 * MAX_NUM_SEQS // 3
    
    list_of_messages = [create_starter_messages(question, index) for index in range(num_seqs)]

    all_extracted_answers = []
    for _ in range(1):
        list_of_messages = batch_message_generate(list_of_messages)
        
        if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            df = pd.DataFrame(
                {
                    "question": [question] * len(list_of_messages),
                    "message": [messages[-1]["content"] for messages in list_of_messages],
                }
            )
            df.to_csv(f"{str(int(time.time() - start_time)).zfill(5)}.csv", index=False)
        
        list_of_messages, extracted_answers = batch_message_filter(list_of_messages)
        all_extracted_answers.extend(extracted_answers)
    
    print(all_extracted_answers)
    answer = select_answer(all_extracted_answers)
    print(answer)

    print("\n\n")
    cutoff_times.pop()
    return answer
#yoink.
if(not is_submission):
    eval_data = [
        # Original 50 problems from the prompt start here
        "A positive integer $n$ is said to be $k$-consecutive if it can be written as the sum of $k$ consecutive positive integers. Find the number of positive integers less than $1000$ that are either $9$-consecutive or $11$-consecutive (or both), but not $10$-consecutive.",
        "An infinite number of buckets, labeled $1$, $2$, $3$, \\ldots, lie in a line. A red ball, a green ball, and a blue ball are each tossed into a bucket, such that for each ball, the probability the ball lands in bucket $k$ is $2^{-k}$. Given that all three balls land in the same bucket $B$ and that $B$ is even, then the expected value of $B$ can be expressed as $\\tfrac{m}{n}$, where $m$ and $n$ are relatively prime positive integers. Find $m+n$.",
        "Let $ABC$ be a triangle with $AB=340$, $BC=146$, and $CA=390$. If $M$ is a point on the interior of segment $BC$ such that the length $AM$ is an integer, then the average of all distinct possible values of $AM$ can be expressed in the form $\\tfrac{p}{q}$, where $p$ and $q$ are relatively prime positive integers. Find $p+q$.",
        "A committee has an oligarchy, consisting of $A\\%$ of the members of the committee. Suppose that $B\\%$ of the work is done by the oligarchy. If the average amount of work done by a member of the oligarchy is $16$ times the amount of work done by a nonmember of the oligarchy, find the maximum possible value of $B-A$.",
        "Over all ordered triples of positive integers $(a,b,c)$ for which $a+b+c^2=abc$, compute the sum of all values of $a^3+b^2+c$.",
        "Let $ABCD$ be a parallelogram such that $AB=40$, $BC=60$, and $BD=50$. Two externally tangent circles of radius $r$ are positioned in the interior of the parallelogram. The largest possible value of $r$ is $\\sqrt{m}-\\sqrt{n}$, where $m$ and $n$ are positive integers. Find $m+n$.",
        "Positive integers $a$, $b$, $c$ satisfy $\\operatorname{lcm}(\\gcd(a,b),c)=180$, $\\operatorname{lcm}(\\gcd(b,c),a)=360$, $\\operatorname{lcm}(\\gcd(c,a),b)=540$. Find the minimum possible value of $a+b+c$.",
        "A number is increasing if its digits, read from left to right, are strictly increasing. For instance, $5$ and $39$ are increasing while $224$ is not. Find the smallest positive integer not expressible as the sum of three or fewer increasing numbers.",
        "A positive integer $x$ is lexicographically smaller than a positive integer $y$ if for some positive integer $i$, the $i$th digit of $x$ from the left is less than the $i$th digit of $y$ from the left, but for all positive integers $j<i$, the $j$th digit of $x$ is equal to the $j$th digit of $y$ from the left. Say the $i$th digit of a positive integer with less than $i$ digits is $-1$. For instance, $11$ is lexicographically smaller than $110$, which is in turn lexicographically smaller than $12$. Let $A$ denote the number of positive integers $m$ for which there exists an integer $n\\ge2020$ such that when the elements of the set $\\{1,2,\\ldots,n\\}$ are sorted lexicographically from least to greatest, $m$ is the $2020$th number in this list. Find the remainder when $A$ is divided by $1000$.",
        "A knight begins on the point $(0,0)$ in the coordinate plane. From any point $(x,y)$ the knight moves to either $(x+2,y+1)$ or $(x+1,y+2)$. Find the number of ways the knight can reach the point $(15,15)$.",
        "At the local Blast Store, there are sufficiently many items with a price of \\$n.99 for each nonnegative integer $n$. A sales tax of $7.5\\%$ is applied on all items. If the total cost of a purchase, after tax, is an integer number of cents, find the minimum possible number of items in the purchase.",
        "In a math competition, all teams must consist of between $12$ and $15$ members, inclusive. Mr. Beluhov has $n>0$ students and he realizes that he cannot form teams so that each of his students is on exactly one team. Find the sum of all possible values of $n$.",
        "There exists a unique positive real number $x$ satisfying$ x=\\sqrt{x^2+\\frac{1}{x^2}}-\\sqrt{x^2-\\frac{1}{x^2}}.$Given that $x$ can be written as $x=2^{m/n}\\cdot3^{-p/q}$ for positive integers $m$, $n$, $p$, and $q$, with $\\gcd(m,n)=\\gcd(p,q)=1$, find $m+n+p+q$.",
        "Let $ABCD$ be a rectangle with sides $AB>BC$ and let $E$ be the reflection of $A$ over $\\overline{BD}$. If $EC=AD$ and the area of $ECBD$ is $144$, find the area of $ABCD$.",
        "Find the number of complex numbers $z$ satisfying $|z|=1$ and $z^{850}+z^{350}+1=0$.",
        "For every positive integer $n$, define$ f(n)=\\frac{n}{1\\cdot3\\cdot5\\cdots (2n+1)}.$Suppose that the sum $f(1)+f(2)+\\cdots+f(2020)$ can be expressed as $\\tfrac{p}{q}$ for relatively prime integers $p$ and $q$. Find the remainder when $p$ is divided by $1000$.",
        "Let $ABCD$ be a cyclic quadrilateral with $AB=6$, $AC=8$, $BD=5$, and $CD=2$. Let $P$ be the point on $\\overline{AD}$ such that $\\angle APB=\\angle CPD$. Then $\\tfrac{BP}{CP}$ can be expressed in the form $\\tfrac{m}{n}$, where $m$ and $n$ are relatively prime positive integers. Find $m+n$.",
        "Let $1=d_1<d_2<\\cdots<d_k=n$ be the positive divisors of a positive integer $n$. Let $S$ be the sum of all positive integers $n$ satisfying$ n=d_1^1+d_2^2+d_3^3+d_4^4.$Find the remainder when $S$ is divided by $1000$.",
        "Let $ABC$ be a triangle with $\\angle ACB=90^\\circ$ and let $r_A$, $r_B$, $r_C$ denote the radii of the excircles opposite $A$, $B$, and $C$, respectively. If $r_A=9$ and $r_B=11$, then $r_C$ can be expressed in the form $m+\\sqrt{n}$, where $m$ and $n$ are positive integers and $n$ is not divisible by the square of any prime. Find $m+n$.",
        "Chris writes on a piece of paper the positive integers from $1$ to $8$ in that order. Then, he randomly writes either $+$ or $\\times$ between every two adjacent numbers, each with equal probability. The expected value of the expression he writes can be expressed as $\\tfrac{p}{q}$ for relatively prime positive integers $p$ and $q$. Find the remainder when $p+q$ is divided by $1000$.",
        "Find the sum of all integer bases $b>9$ for which $17_b$ is a divisor of $97_b$.",
        "In $\\triangle ABC$ points $D$ and $E$ lie on $\\overline{AB}$ so that $AD<AE<AB$, while points $F$ and $G$ lie on $\\overline{AC}$ such that $AF<AG<AC$. Suppose $AD=4$, $DE=16$, $EB=8$, $AF=13$, $FG=52$, and $GC=26$. Let $M$ be the reflection of $D$ through $F$, and let $N$ be the reflection of $G$ through $E$. The area of quadrilateral $DEGF$ is $288$. Find the area of heptagon $AFNBCEM$.",
        "The $9$ members of a baseball team went to an ice-cream parlor after their game. Each player had a single scoop cone of chocolate, vanilla, or strawberry ice cream. At least one player chose each flavor, and the number of players who chose chocolate was greater than the number who chose vanilla, which in turn was greater than the number who chose strawberry. Let $N$ be the number of different assignments of flavors to players that meet these conditions. Find the remainder when $N$ is divided by $1000$.",
        "Find the number of ordered pairs $(x,y)$, where both $x$ and $y$ are integers between $-100$ and $100$ inclusive, such that $12x^2-xy-6y^2=0$.",
        "There are $8!=40320$ eight-digit positive integers that use each of the digits $1,2,3,4,5,6,7,8$ exactly once. Let $N$ be the number of these integers that are divisible by $22$. Find the absolute difference between $N$ and $2025$.",
        "An isosceles trapezoid has an inscribed circle tangent to each of its four sides. The radius of the circle is $3$, and the area of the trapezoid is $72$. Let the parallel sides of the trapezoid have lengths $r$ and $s$, with $r\\neq s$. Find $r^2+s^2$.",
        "The twelve letters $A$, $B$, $C$, $D$, $E$, $F$, $G$, $H$, $I$, $J$, $K$, and $L$ are randomly grouped into six pairs. The two letters in each pair are placed in alphabetical order to form a two‐letter word, and then those six words are listed alphabetically. The probability that the last word listed contains $G$ is $\\tfrac{m}{n}$, where $m$ and $n$ are relatively prime positive integers. Find $m+n$.",
        "Let $k$ be a real number such that the system \\begin{align*} &|25+20i-z|=5 \\\\ &|z-4-k|=|z-3i-k| \\end{align*} has exactly one complex solution $z$. The sum of all possible values of $k$ can be written as \\(\\frac{m}{n}\\), where $m$ and $n$ are relatively prime positive integers. Find $m+n$. Here $i=\\sqrt{-1}$.",
        "The parabola with equation $y=x^2-4$ is rotated $60^\\circ$ counterclockwise around the origin. The unique point in the fourth quadrant where the original parabola and its image intersect has $y$-coordinate $\\frac{a-\\sqrt{b}}{c}$, where $a$, $b$, and $c$ are positive integers, and $a$ and $c$ are relatively prime. Find $a+b+c$.",
        "The $27$ cells of a $3\\times9$ grid are filled using the numbers $1$ through $9$ so that each row contains $9$ different numbers, and each of the three $3\\times3$ blocks (heavily outlined as in the first three rows of a Sudoku puzzle) contains $9$ different numbers. The number of different ways to fill such a grid can be written as $p^a\\cdot q^b\\cdot r^c\\cdot s^d$, where $p$, $q$, $r$, and $s$ are distinct primes and $a$, $b$, $c$, $d$ are positive integers. Find $p\\cdot a+q\\cdot b+r\\cdot c+s\\cdot d$.",
        "A piecewise linear function is defined by$ f(x)=\\begin{cases} x&\\text{if } -1\\le x<1 \\\\ 2-x&\\text{if } 1\\le x<3 \\end{cases} $and $f(x+4)=f(x)$ for all real $x$. The graph of $f(x)$ has a sawtooth pattern. The parabola $x=34y^2$ intersects $f(x)$ at finitely many points. The sum of the $y$-coordinates of these intersection points can be expressed as $\\tfrac{a+b\\sqrt{c}}{d}$, where $a$, $b$, $c$, and $d$ are positive integers with $\\gcd(a,b,d)=1$, and $c$ is squarefree. Find $a+b+c+d$.",
        "The set of points in $3$-dimensional space lying on the plane $x+y+z=75$ and satisfying$ x-yz<y-zx<z-xy $forms three disjoint convex regions; exactly one of these regions has finite area. The area of that region can be written as $a\\sqrt{b}$, where $a$ and $b$ are positive integers and $b$ is squarefree. Find $a+b$.",
        "Alex divides a disk into four quadrants with two perpendicular diameters through its center. He then draws $25$ additional line segments by selecting two points at random on the perimeter in different quadrants and connecting them. Find the expected number of regions the $27$ segments form.",
        "Let $ABCDE$ be a convex pentagon with $AB=14$, $BC=7$, $CD=24$, $DE=13$, and $EA=26$, with $\\angle B=60^\\circ$. For each point $X$ in the plane, define $f(X)=AX+BX+CX+DX+EX$. The minimum value of $f(X)$ can be expressed as $m+n\\sqrt{p}$, where $m$, $n$, and $p$ are positive integers and $p$ is squarefree. Find $m+n+p$.",
        "Let $N$ denote the number of ordered triples of positive integers $(a,b,c)$ such that $a,b,c\\le3^6$ and $a^3+b^3+c^3$ is divisible by $3^7$. Find the remainder when $N$ is divided by $1000$.",
        "Six points $A$, $B$, $C$, $D$, $E$, and $F$ lie in a line in that order. Suppose a point $G$ (not on the line) and the distances $AC=26$, $BD=22$, $CE=31$, $DF=33$, $AF=73$, $CG=40$, and $DG=30$ are given. Find the area of $\\triangle BGE$.",
        "Find the sum of all positive integers $n$ such that $n+2$ divides the product $3(n+3)(n^2+9)$.",
        "Four unit squares form a $2\\times2$ grid. Each of the $12$ unit segments on the grid is colored either red or blue so that each square has exactly $2$ red and $2$ blue sides. Find the number of such colorings.",
        "The product $ \\prod_{k=4}^{63} \\frac{\\log_k (5^{k^2-1})}{\\log_{k+1} (5^{k^2-4})} $ equals $\\tfrac{m}{n}$, where $m$ and $n$ are relatively prime. Find $m+n$.",
        "Suppose $\\triangle ABC$ has angles $84^\\circ$, $60^\\circ$, and $36^\\circ$. Let $D$, $E$, and $F$ be the midpoints of $BC$, $AC$, and $AB$, respectively. The circumcircle of $\\triangle DEF$ intersects $BD$, $AE$, and $AF$ at $G$, $H$, and $J$ respectively. Find $\\overarc{DE}+2\\cdot\\overarc{HJ}+3\\cdot\\overarc{FG}$ (in degrees).",
        "Circle $\\omega_1$ with radius $6$ centered at $A$ is internally tangent at $B$ to circle $\\omega_2$ with radius $15$. Points $C$ and $D$ lie on $\\omega_2$ with $\\overline{BC}$ as a diameter and $\\overline{BC}\\perp\\overline{AD}$. A rectangle $EFGH$ is inscribed in $\\omega_1$ so that $EF\\perp BC$, with $C$ closer to $GH$ than to $EF$ and $D$ closer to $FG$ than to $EH$. If triangles $DGF$ and $CHG$ have equal area and the area of $EFGH$ is $\\frac{m}{n}$ (with $m,n$ coprime), find $m+n$.",
        "Let $A$ be the set of positive divisors of $2025$, and let $B$ be a randomly chosen subset of $A$. The probability that $B$ is nonempty and its least common multiple is $2025$ is $\\frac{m}{n}$, with $m$ and $n$ coprime. Find $m+n$.",
        "From an unlimited supply of 1-cent, 10-cent, and 25-cent coins, Silas wishes to form a total of $N$ cents using the greedy algorithm. (For instance, to make 42 cents he picks 25, then 10, then 7 ones.) The greedy algorithm succeeds if no other coin combination uses fewer coins than the greedy choice. Find the number of values of $N$ between $1$ and $1000$ for which the algorithm succeeds.",
        "There are $n$ values of $x$ in $(0,2\\pi)$ where $f(x)=\\sin(7\\pi\\sin(5x))=0$, and for $t$ of those the graph is tangent to the $x$-axis. Find $n+t$.",
        "Sixteen chairs are arranged in a row. Eight people choose chairs such that no one sits next to two other people. Let $N$ be the number of chair subsets possible. Find the remainder when $N$ is divided by $1000$.",
        "Let $S$ be the set of vertices of a regular 24-gon. Find the number of ways to draw 12 equal segments such that each vertex belongs to exactly one segment.",
        "Let $A_1A_2\\dots A_{11}$ be a nonconvex 11-gon where the area of $A_iA_1A_{i+1}$ is 1 for $2\\le i\\le10$, $\\cos(\\angle A_iA_1A_{i+1})=\\frac{12}{13}$, and the perimeter is 20. If $A_1A_2+A_1A_{11}$ can be expressed as $\\frac{m\\sqrt{n}-p}{q}$ (with $n$ squarefree and $m,p,q$ coprime), find $m+n+p+q$.",
        "Let the sequence of rationals $x_1,x_2,\\dots$ be defined by $x_1=\\frac{25}{11}$ and$ x_{k+1}=\\frac{1}{3}\\Bigl(x_k+\\frac{1}{x_k}-1\\Bigr).$ Express $x_{2025}$ as $\\frac{m}{n}$ (with $m,n$ coprime) and find the remainder when $m+n$ is divided by $1000$.",
        "Let $\\triangle ABC$ be a right triangle with $\\angle A=90^\\circ$ and $BC=38$. There exist points $K$ and $L$ inside the triangle such that $AK=AL=BK=CL=KL=14$. The area of quadrilateral $BKLC$ can be written as $n\\sqrt{3}$; find $n$.",
        "Let $ f(x)=\\frac{(x-18)(x-72)(x-98)(x-k)}{x}. $ There exist exactly three positive real values of $k$ for which $f$ attains a minimum at exactly two distinct $x$ values. Find the sum of these three $k$ values.",
        # 25 Modified PDF problems start here
        "Call a 9-digit number a cassowary if it uses each of the digits 1 through 9 exactly once. Compute the number of cassowaries that are prime.",
        "Compute $V = \\frac{20+\\frac{1}{25-\\frac{1}{20}}}{25+\\frac{1}{20-\\frac{1}{25}}}$. If $V=m/n$ for coprime positive integers $m,n$, find $m+n$.",
        "Jacob rolls two fair six-sided dice. If the outcomes of these dice rolls are the same, he rolls a third fair six-sided die. The probability that the sum of outcomes of all the dice he rolls is even can be expressed as $m/n$ for coprime positive integers $m,n$. Find $m+n$.",
        "Let $\\triangle ABC$ be an equilateral triangle with side length 4. Across all points $P$ inside triangle $\\triangle ABC$ satisfying $[PAB] + [PAC] = [PBC]$, compute the minimum possible value of $PA^2$. (Here, $[XYZ]$ denotes the area of triangle $XYZ$.)",
        "Let $r$ be the largest possible radius of a circle contained in the region defined by $|x+|y|| \\le 1$ in the coordinate plane. Compute $(r+2)^2$.",
        "Let $\\triangle ABC$ be an equilateral triangle. Point $D$ lies on segment $BC$ such that $BD=1$ and $DC=4$. Points $E$ and $F$ lie on rays $AC$ and $AB$, respectively, such that $D$ is the midpoint of $EF$. Compute $EF^2$.",
        "The number $\\frac{9^9-8^8}{1001}$ is an integer. Compute the sum of its prime factors.",
        "A checkerboard is a rectangular grid of cells colored black and white such that the top-left corner is black and no two cells of the same color share an edge. Two checkerboards are distinct if and only if they have a different number of rows or columns. Compute the number of distinct checkerboards that have exactly 41 black cells.",
        "Let $P$ and $Q$ be points selected uniformly and independently at random inside a regular hexagon $ABCDEF$. The probability that segment $PQ$ is entirely contained in at least one of the quadrilaterals $ABCD, BCDE, CDEF, DEFA, EFAB,$ or $FABC$ can be expressed as $m/n$ for coprime positive integers $m,n$. Find $m+n$.",
        "A square of side length 1 is dissected into two congruent pentagons. Let $P$ be the least upper bound of the perimeter of one of these pentagons. Compute $(P-2)^2$.",
        "Let $f(n) = n^2 + 100$. Let $f^{(k)}(n)$ denote the result of applying $f$ $k$ times, starting with $n$. Compute the remainder when $f^{(2025)}(1)$ is divided by $1000$.",
        "Holden has a collection of polygons. He writes down a list containing the measure of each interior angle of each of his polygons. He writes down the list $30^\\circ, 50^\\circ, 60^\\circ, 70^\\circ, 90^\\circ, 100^\\circ, 120^\\circ, 160^\\circ$, and $x^\\circ$, in some order. Compute $x$.",
        "A number is upwards if its digits in base 10 are nondecreasing when read from left to right. Compute the number of positive integers less than $10^6$ that are both upwards and multiples of 11.",
        "A parallelogram $P$ can be folded over a straight line so that the resulting shape is a regular pentagon with side length 1. Let $L$ be the perimeter of $P$. Compute $(L-5)^2$.",
        "Right triangle $\\triangle DEF$ with $\\angle D = 90^\\circ$ and $\\angle F = 30^\\circ$ is inscribed in equilateral triangle $\\triangle ABC$ such that $D$, $E$, and $F$ lie on segments $BC$, $CA$, and $AB$, respectively. Given that $BD = 7$ and $DC = 4$, compute $DE^2$.",
        "The Cantor set is defined as the set of real numbers $x$ such that $0 \\le x \\le 1$ and the digit 1 does not appear in the base-3 expansion of $x$. Two numbers are uniformly and independently selected at random from the Cantor set. The expected value of their absolute difference can be expressed as $m/n$ for coprime positive integers $m,n$. Find $m+n$.",
        "Let $f$ be a quadratic polynomial with real coefficients, and let $g_1, g_2, g_3, \\dots$ be a geometric progression of real numbers. Define $a_n = f(n) + g_n$. Given that $a_1, a_2, a_3, a_4, a_5$ are equal to $1, 2, 3, 14, 16$, respectively, the ratio $g_2/g_1$ can be expressed as $m/n$ for coprime positive integers $m,n$. Find $m+n$.",
        "Let $f: \\{1,2,3,\\dots,9\\} \\to \\{1, 2, 3, \\dots ,9\\}$ be a permutation chosen uniformly at random from the $9!$ possible permutations. Let $f^{(k)}(n)$ denote the result of applying $f$ $k$ times, starting at $n$. The expected value of $f^{(2025)}(1)$ can be expressed as $m/n$ for coprime positive integers $m,n$. Find $m+n$.",
        "A subset $S$ of $\\{1,2,3,\\dots,2025\\}$ is called balanced if for all elements $a$ and $b$ both in $S$, there exists an element $c$ in $S$ such that $2025$ divides $a + b - 2c$. Compute the number of nonempty balanced subsets, modulo 1000.",
        "Find the 100th smallest positive integer multiple of 7 such that all of its digits in base 10 are strictly less than 3. Compute the remainder when this number is divided by 1000.",
        "Compute the unique 5-digit positive integer $N = \\underline{abcde}$ (where $a,b,c,d,e$ are digits, $a \\ne 0, c \\ne 0$) such that $N = (\\underline{ab} + \\underline{cde})^2$. Here $\\underline{xy}$ denotes the integer $10x+y$ and $\\underline{xyz}$ denotes $100x+10y+z$. Compute the remainder when $N$ is divided by 1000.",
        "Let $a, b, c$ be real numbers such that $a^2(b + c) = 1$, $b^2(c + a) = 2$, and $c^2(a + b) = 5$. It is given that there are three possible values for the product $abc$. Let $V$ be the minimum possible value of $abc$. Compute $(2V+5)^2$.",
        "A regular hexagon $ABCDEF$ has side length 2. A circle $\\omega$ lies inside the hexagon and is tangent to segments $AB$ and $AF$. There exist two perpendicular lines tangent to $\\omega$ that pass through $C$ and $E$, respectively. Given that these two lines do not intersect on line $AD$, let $r$ be the radius of $\\omega$. Compute $(2r+3)^2$.",
        "For any integer $x$, let $f(x) = 100! \\left(1 + x + \\frac{x^2}{2!} + \\dots + \\frac{x^{100}}{100!}\\right)$. A positive integer $a$ is chosen such that $f(a) - 20$ is divisible by $101^2$. Compute the remainder when $f(a + 101)$ is divided by 1000.",
        "Let $ABCD$ be a trapezoid such that $AB \\parallel CD$, $AD = 13$, $BC = 15$, $AB = 20$, and $CD = 34$. Point $X$ lies inside the trapezoid such that $\\angle XAB = 2\\angle XBA$ and $\\angle XDC = 2\\angle XCD$. Compute $XD-XA$."
    ]

    answers = [
        # Original 50 answers from the prompt start here
        "165", "191", "366", "60", "225", "93", "788", "443", "680", "252",
        "20", "480", "14", "192", "433", "436", "847", "680", "847", "302",
        "70", "588", "16", "117", "279", # Changed from N-2025 to |N-2025| in problem text
        "504", "821", "77", "62", "81",
        "259", "510", "204", "60", "735", "468", "49", "82", "106", "336",
        "293", "237", "610", "149", "907", "113", "19", "248", "104", "240",
        # 25 Modified PDF answers start here
        "0",   # Problem 1
        "9",   # Problem 2 (4+5)
        "17",  # Problem 3 (5+12)
        "3",   # Problem 4 (sqrt(3)^2)
        "8",   # Problem 5 ((2sqrt(2)-2+2)^2)
        "52",  # Problem 6 ((2sqrt(13))^2)
        "231", # Problem 7
        "9",   # Problem 8
        "11",  # Problem 9 (5+6)
        "18",  # Problem 10 ((2+3sqrt(2)-2)^2)
        "101", # Problem 11 (3101 mod 1000)
        "220", # Problem 12
        "219", # Problem 13
        "5",   # Problem 14 ((5+sqrt(5)-5)^2)
        "13",  # Problem 15 (sqrt(13)^2)
        "7",   # Problem 16 (2+5)
        "29",  # Problem 17 (19+10)
        "9",   # Problem 18 (7+2)
        "751", # Problem 19 (3751 mod 1000)
        "221", # Problem 20 (221221 mod 1000)
        "209", # Problem 21 (88209 mod 1000)
        "5",   # Problem 22 ((2*(-5-sqrt(5))/2+5)^2)
        "27",  # Problem 23 ((2*(3sqrt(3)-3)/2+3)^2)
        "939", # Problem 24 (1939 mod 1000)
        "4"    # Problem 25
    ]
    s = ""
    num_correct = 0

    total_questions = 0
    if(part == 1):
        start_idx = 0
        end_idx = 38
    else:
        start_idx = 38
        end_idx = 75
    for idx in range(start_idx, end_idx):
        question = eval_data[idx]
        prediction = predict_for_question(question)
        print(f"Prediction for q{idx}: {prediction}")
        total_questions += 1
        if prediction == int(answers[idx]):
            s += "C"
            num_correct += 1
        else:
            s += "X"
        print(f"Accuracy: {num_correct / total_questions}")
    print(s)


if is_submission:
    def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
        id_ = id_.item(0)
        print("="*100)
        print(id_)
        start_time = time.time()
        question = question.item(0)
        print("question: ", question)
        answer = predict_for_question(question)
        end_time = time.time()
        print(f"time taken: {end_time - start_time}")
        print("="*100)
        print("predicted answer: ", answer)
        return pl.DataFrame({'id': id_, 'answer': answer})
    pd.read_csv(
        '/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv'
    ).drop('answer', axis=1).to_csv('reference.csv', index=False)
    inference_server = kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer(predict)
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        inference_server.serve()
    else:
        inference_server.run_local_gateway(
            (
                'reference.csv',
            )
        )
