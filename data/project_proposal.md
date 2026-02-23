\documentclass{article}

\usepackage[final]{neurips_2019}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{hyperref}
\usepackage{url}
\usepackage{booktabs}
\usepackage{amsfonts}
\usepackage{nicefrac}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{lipsum}

\usepackage{tcolorbox}

\usepackage{soul}   % in preamble
\sethlcolor{yellow} % optional, to set highlight color

\newcommand{\note}[1]{\textcolor{blue}{{#1}}}

\title{
  Structure-Aware Contrastive Learning for Fine-Grained Math Standard Alignment \\
  \vspace{1em}
  \small{\normalfont Stanford CS224N \textbf{Custom} Project}  % Select one and delete the other
}

\author{
  Xinman Liu, Teah Shi, Mayank Sharma \\
  Graduate School of Education\\
  Stanford University \\
  \texttt{\{xinman, teah2001, masharma\}@stanford.edu}
 \\
  % Examples of more authors
%   \And
%   Name \\
%   Department of Computer Science \\
%   Stanford University \\
%   \texttt{name@stanford.edu} \\
%   \And
%   Name \\
%   Department of Computer Science \\
%   Stanford University \\
%   \texttt{name@stanford.edu}
}

\begin{document}

\maketitle

% \begin{abstract}
%   Required for final report
% \end{abstract}




\section{Key Information}

\begin{itemize}
    \item External collaborators: None
    \item Mentor: We have no particular mentor.
    \item Sharing project: No
\end{itemize}


\section{Research paper summary (max 2 pages)}

\begin{table}[h]
    \centering
    \begin{tabular}{ll}
        \toprule
        \textbf{Title} & MathFish: Evaluating Language Model Math Reasoning via Grounding in Educational Curricula \\
        \midrule
         \textbf{Authors} & Lucy Li, Tal August, Rose E. Wang, Luca Soldaini, Courtney Allison, and Kyle Lo \\
        \textbf{Venue} & Findings of the Association for Computational Linguistics, EMNLP \\
        \textbf{Year}  & 2024 \\
        \textbf{URL}   & \url{https://aclanthology.org/2024.findings-emnlp.323.pdf} \\
        \bottomrule
    \end{tabular}
    \vspace{1em}
    \caption{Bibliographical information~\cite{li2024mathfish}.}
\end{table}

\paragraph{Background.} Most existing benchmarks for mathematical reasoning in language models focus on a single question: can the model get the right answer? Datasets such as GSM8k (\cite{cobbe2021gsm8k}) and MATH (\cite{hendrycks2021measuring}) test whether a model can solve a problem, but they rarely examine what mathematical competencies that problem actually exercises. But in actual K–12 education, the categorization of math content matters more than it might seem. Professional curriculum reviewers spend months mapping published math problems to fine-grained pedagogical standards, for instance, skills like "multiplication procedures for fractions" or concepts like "understanding area versus volume." So rather than asking whether LMs can answer math questions correctly, the MathFish paper asks whether they can identify what skills and concepts a student would learn or practice by completing those questions. The authors argue that this angle of evaluation is both novel and necessary, because as LMs are increasingly adopted in classrooms for generating assessments, lesson plans, and tutoring dialogues, we need to know whether these models understand the pedagogical structure behind math content, not just its surface-level semantics.

\paragraph{Summary of contributions.} The paper makes three main contributions. First, it introduces two datasets grounded in real-world educational practice: Achieve the Core (ATC), containing 385 fine-grained K–12 math standards from the Common Core framework, organized in a hierarchy of grades \& domains, clusters, and individual standards, along with 1,040 conceptual connections between them; MathFish then provides 9,900 math problems scraped from two reputable open educational resources (Illustrative Mathematics and Fishtank Learning), each labeled by publishers with aligned Common Core standards.

Second, the authors design two task formats for probing LM understanding of curriculum alignment. In verification, a model receives a single problem paired with a single standard and must judge whether the problem fully aligns with that standard (a binary yes/no decision). In tagging, a model is given a problem and must traverse a hierarchical decision tree, first selecting the appropriate domain, then cluster, then specific standard(s) that the problem teaches. These formats mirror how curriculum reviewers actually work: sometimes confirming a publisher's claimed alignment, other times labeling materials from scratch. The authors test GPT-4-Turbo, Mixtral-8x7B, and Llama-2-70B across zero-shot, one-shot, and three-shot prompting setups. For stress-testing, the authors pair each problem not only with its true aligned standard but also with deliberately chosen unaligned standards that vary in how "close" they are to the correct one. Some distractors come from a completely different math domain and grade level (easy to rule out); others share the same domain and grade, or are direct conceptual neighbors of the correct standard in ATC's coherence map (much harder to distinguish). The results confirm what you would expect: accuracy drops sharply as the distractor moves closer to the true standard in the curriculum hierarchy. Three-shot GPT-4 performs best overall, yet it still struggles once the unaligned standard is a same-cluster sibling or a conceptual neighbor. In tagging, GPT-4 achieves only about 5\% exact-match accuracy when navigating the full hierarchy on its own, though its predictions tend to land conceptually close to the ground truth (average 1.9 edges away from the gold standard in the coherence map, compared to 5.5 for random guessing).

Third, the paper proposes two case studies that ground these tasks in practical settings. In one, the authors prompt LMs to generate math problems targeting specific standards, then ask experienced K–12 teachers to judge whether each generated problem actually aligns with the standard it was supposed to address. They also run the GPT-4 verifier from the earlier experiments on these same generated problems for comparison. The results show that GPT-4 judges 96\% of its own generated problems as fully aligned with the intended standard, but teachers rate only 52\% that way. In the other case study, the authors use their best tagging setup to label every problem in GSM8k with Common Core standards. The results show that GSM8k covers only about a third of all K–12 standards, concentrated heavily in arithmetic operations. More interestingly, when the authors cross-reference these tags with models' problem-solving accuracy on GSM8k, problems tagged with higher-grade standards have consistently lower solve rates, and this holds for every model family and size tested.

\paragraph{Limitations and discussion.} The most obvious limitation is the text-only setup. All images, web applets, and interactive elements in the original curricula get replaced with a dummy token. For standards involving geometry, graphing, or data visualization, this likely understates what models could do with multimodal input, or alternatively overstates it for standards where visual reasoning is essential. The authors acknowledge this gap and flag it for future work.

Additionally, the teacher annotations come from a single reviewing organization (EdReports), and each annotator made a single pass over the materials rather than engaging in the kind of deliberative consensus typical of real curriculum review. The annotator pool is also narrow: mostly White women based in the Midwest, raises questions about whether alignment judgments would hold across more diverse educator populations. Additionally, the paper's reliance on Common Core State Standards restricts applicability to the U.S. context. Educators operating under different national or state frameworks may find the findings less transferable.

On the modeling side, the paper evaluates only prompting-based approaches and does not explore fine-tuning or training-based methods. This is a reasonable scope decision for an evaluation paper, but it leaves open the question of how much room for improvement exists when models can learn from the rich structural relationships in the ATC hierarchy. The error analysis shows that model mistakes are not random but cluster around structurally nearby standards (siblings in the same cluster, conceptual neighbors, adjacent grades). This implies that a training-based method which explicitly leverages the hierarchy and connection structure in ATC might do considerably better than 5\% exact match.

\paragraph{Why this paper?} We chose MathFish because it speaks directly to our research interest in the gap between what LMs approximate and what educators actually need. Coming from education backgrounds, we have repeatedly encountered situations where LLMs appear highly capable on the surface, yet fail in ways that matter deeply for teaching and learning. The paper’s central question, whether models can identify the pedagogical skills and concepts that math problems are intended to teach, resonated strongly with our experience. In a domain like curriculum design, that gap between pattern-matching and genuine understanding is exactly where things break down. Having read the paper in depth, we do feel that it delivered what we were hoping for. The dataset is grounded in real curriculum standards, and the analysis honestly shows where current models fall short. In particular, the structured error patterns it documents gave us useful perspective and direction for our own approach.

\paragraph{Wider research context.} The paper is relevant to the wider research context of the growing intersection of NLP and education, where domain expertise (e.g., curriculum design) provides structured knowledge that could improve model training and evaluation.

First, it investigates the opportunities and limitations of general-purpose LMs when applied to vertical domain challenges: in this case, recognizing pedagogical and curricular boundaries. For instance, a problem about multiplying by 10 might align with either “4.NBT.A.1: Recognize that in a multi-digit number, a digit represents 10× what it represents in the place to its right” or “4.NBT.B.5: Multiply a whole number of up to four digits by a one-digit whole number.” However, while both involve multiplication by 10, the pedagogical \textit{intent} differs: the former focuses on the \textit{conceptual} understanding on place value (i.e. knowing why), whereas the latter focuses on the the \textit{procedural} understanding of multiplication (i.e. knowing how). This distinction parallels challenges in representation learning, such as whether models capture not just semantic similarity but also pragmatic and contextual differences.

Second, the hierarchical tagging task represents a structured prediction problem where labels (standards) have explicit relationships (graph edges, hierarchical organization), connecting to dependency parsing and other tasks where output structure matters beyond simply approximating semantic neighborhoods. The challenge here is to learn precise boundaries rather than coarse clustering.

Lastly, the paper exemplifies the broader trend in NLP toward ecologically valid evaluation, for instance, extrinsic evaluation of models on downstream tasks that matter to real users (e.g., teachers) rather than artificial benchmarks that may incentivize narrow optimization misaligned with real-world problem spaces.

\section{Project description (1-2 pages)}

\paragraph{Goal.} \textit{Question}: Can we substantially improve exact standard-level alignment accuracy on the MathFish dataset by encoding curriculum structure directly into the model's training objective through contrastive learning with pedagogically-informed hard negatives?

\textit{Why this matters}: Current state-of-the-art models (e.g., GPT-4) achieve only ~5\% exact-match accuracy on MathFish tagging without hierarchy hints. They often select nearby standards but miss the precise pedagogical intent, highlighting that the challenge lies in distinguishing adjacent concepts, not in basic semantic understanding. As LMs are increasingly used for curriculum generation and assessment, improving their grasp of these educational distinctions is both scientifically and practically important.

\textit{Why it should work}: Errors are structured: most incorrect predictions are siblings or conceptually connected standards. This predictable confusion can be learned and exploited to improve alignment.

\textit{Relation to MathFish}: We build directly on the MathFish dataset and evaluation protocol, moving from prompting-based evaluation to training-based improvement, enabling direct performance comparison.



\paragraph{Task.} 

The task is multi-label curriculum standards alignment without hints about their hierarchies.

\textit{Input}: A math word problem in natural language \\
\textit{Output}: The complete set of Common Core math standards (from 385 possible standards) that the problem aligns with.



Unlike the paper's assisted tagging experiments (where models traverse a hierarchy with correct branches revealed), we tackle the hardest setting: selecting from all 385 standards simultaneously. This mirrors real-world use cases where educators need to tag arbitrary problems without hints.

\begin{tcolorbox}[title=Example, colback=gray!5, colframe=black, boxrule=0.8pt]
\textbf{Input problem}
\begin{quote}
Kipton has a digital scale. He puts a marshmallow on the scale and it reads 7.2 grams. How much would you expect 10 marshmallows to weigh? Why?
\end{quote}
\textbf{Output standards (abbreviated)}
\begin{itemize}
    \item Explain patterns in the number of zeros when multiplying by powers of 10, and explain patterns in decimal point placement.
    \item Recognize that in a multi-digit number, a digit in one place represents 10 times as much as it represents in the place to its right.
\end{itemize}
\end{tcolorbox}


\paragraph{Data.}


We will use MathFish (Lucy et al., 2024), a dataset containing approximately 9,900 math problems from Illustrative Mathematics and Fishtank Learning, labeled with fine-grained Common Core standards. The 385 standards are organized hierarchically (grade → domain → cluster → standard) with 1,040 conceptual connections between them. Natural language descriptions are provided for each standard. We will use the paper's evaluation split (20\% of data) for testing. We will further split the training data into 80\% train / 20\% validation.

For preprocessing, we will: (1) clean and normalize problem text by removing  HTML artifacts and standardizing table formatting, (2) construct a curriculum  graph from the ATC hierarchy and conceptual connections to enable hard  negative sampling, (3) for each training problem, dynamically sample hard  negative standards including sibling standards within the same cluster, conceptually connected but misaligned standards, and grade-adjacent standards  (±1 level), and (4) create balanced training batches with one positive example and 3–5 hard negatives per problem.




\paragraph{Methods.}


Our system has two core contributions:\\

\textbf{Component 1: Curriculum Structure-Aware Contrastive Learning.} We train a bi-encoder to embed problems and standards in a shared space using contrastive learning with curriculum-informed hard negatives. The encoder uses a pre-trained sentence transformer (all-mpnet-base-v2) with separate 256-dim projection heads and L2-normalized embeddings. The InfoNCE loss is applied with temperature scaling. Hard negatives are sampled from siblings (40\%), conceptually connected standards (30\%), grade-adjacent standards (20\%), and random negatives (10\%). Unlike standard contrastive learning, we weight negatives by pedagogical distance to penalize nearby-but-wrong standards more heavily. Implementation uses PyTorch and Hugging Face Transformers with AdamW optimizer.  

\textbf{Component 2: Cross-Encoder Re-ranking.} The bi-encoder retrieves top-$k$ candidates ($k=20$), which are re-scored using a BERT or RoBERTa cross-encoder with a binary classification head. Positive examples are gold alignments; negatives are top-$k$ non-aligned standards. Binary cross-entropy loss is used for fine-tuning. This two-stage design balances efficiency (bi-encoder over 385 standards) and accuracy (cross-encoder for detailed interaction). 

\textit{Originality.} The main innovations are curriculum-informed hard negative sampling and the two-stage bi-encoder + cross-encoder design targeting structured alignment errors. All components are implemented by us using existing pretrained models where indicated.

\paragraph{Baselines.}


\begin{itemize}
    \item \textbf{MathFish reported results:} Zero-shot and three-shot GPT-4 prompting (~5\% exact match), cited directly from the paper.
    
    \item \textbf{Bi-encoder without contrastive training (ours):} Off-the-shelf 
    sentence transformer (all-mpnet-base-v2) with nearest-neighbor retrieval; 
    isolates the effect of contrastive learning.
    
    \item \textbf{Bi-encoder with random negative sampling (ours):} Same 
    architecture as Component 1, but with random negatives instead of 
    curriculum-aware hard negatives; isolates the contribution of 
    pedagogically-informed sampling.
    
    \item \textbf{Bi-encoder with curriculum-aware training only (ours):} 
    Our Component 1 (contrastive bi-encoder with hard negatives) without 
    Component 2 (cross-encoder re-ranking); tests whether re-ranking provides 
    additional gains.
    
    \item \textbf{Flat multi-label classifier (ours):} BERT encoder + 385-way 
    classification head; tests the necessity of the two-stage retrieval and 
    re-ranking design.
\end{itemize}

All implemented baselines use the same train/validation/test splits and hyperparameter search budget for fair comparison.


\paragraph{Evaluation.}


We will evaluate using \textbf{exact match accuracy} as our primary metric, 
where the predicted set of standards must exactly match the gold set (~5\% 
baseline in MathFish). Secondary metrics include: \textbf{Micro-/Macro-F1} 
for partial credit, \textbf{Recall@k ($k=5, 10, 20$)} for the bi-encoder 
retrieval stage, \textbf{average graph distance} in the ATC hierarchy between 
predicted and gold standards, and \textbf{sibling confusion rate} (percentage 
of errors where the predicted standard is a same-cluster sibling of the correct 
standard). 

Our performance targets are: (1) 15–20\% exact match accuracy (3–4× improvement 
over baseline), (2) Recall@20 > 75\% (ensuring correct standards reach 
re-ranking stage), (3) average graph distance <2.0 edges (better than GPT-4's 
1.9 reported in MathFish), and (4) 30–40\% reduction in sibling confusion 
errors. These targets are reasonable because the MathFish paper shows most 
errors are near-misses within the curriculum structure, which our contrastive 
training with curriculum-aware hard negatives directly addresses. 

We will compare directly against MathFish baselines using their evaluation 
protocol. Ablation studies will examine: (1) contrastive learning vs. 
off-the-shelf embeddings, (2) curriculum-aware vs. random hard negatives, 
(3) impact of different negative sampling ratios (exploring the 40/30/20/10 
distribution), (4) bi-encoder only vs. bi-encoder with cross-encoder 
re-ranking, and (5) effect of $k$ (number of retrieval candidates). 
Qualitative evaluation will include error analysis by standard type 
(procedural vs. conceptual, grade level) and case studies demonstrating 
where curriculum structure improves alignment predictions.

\paragraph{Ethical Implications.} 
Our project raises two primary ethical concerns. First, over-reliance on automated alignment systems could misrepresent what students are expected to learn, leading to poor instructional decisions. For example, a problem incorrectly tagged as assessing “decimal operations” instead of “place value understanding” could result in gaps in prerequisite knowledge, disproportionately affecting students already struggling with math. Second, bias in educational standards and cultural assumptions means that models trained on Common Core frameworks may reinforce specific pedagogical norms and problem types that do not generalize to other contexts or culturally responsive teaching practices, limiting usefulness for diverse educational settings. To mitigate these risks, we frame our system as an assistive tool rather than a replacement, ensure human oversight in alignment decisions, conduct error analysis, release our code and models with clear documentation and limitations. These strategies recognize that technical improvements do not automatically translate to better educational outcomes, emphasizing that curriculum alignment should remain a collaborative human-AI task.

\bibliographystyle{unsrt}
\bibliography{references}

\end{document}
