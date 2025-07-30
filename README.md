# Auto-Probe Framework

This documentation presents comprehensive pseudocode algorithms for all framework components, supplemented by key implementation files that demonstrate the core innovations. The pseudocode ensures full algorithmic transparency for reproducibility, while complete source code, pipeline scripts and environment setup will be made available upon paper acceptance.

Note: Make sure to complete all `[necessary]` fields in the /experiments/pipline_llava1.6.yaml configuration file prior to execution.

Please download:
- https://huggingface.co/google/owlv2-large-patch14-ensemble
- https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b

```bash
conda env create -f environment.yaml

conda activate auto

chmod +x pipeline_llava1.6.sh

./pipeline_llava1.6.sh
```

## Framework Pseudocode

### 1. Hallucination Induction

The framework actively induces hallucination behaviors through strategic prompt modification, creating a temporary modified behavior in the model with increased hallucination tendency.

```
Algorithm 2: Hallucination Induction
Require: Image dataset I, LVLM model M, prompt sets P, generation parameters
Ensure: Generated descriptions D for each image

1: function HALLUCINATIONINDUCTION(I, M, P)
2:    D ← ∅
3:    processor ← LoadProcessor(M)
4:    model ← LoadModel(M)
5:    
6:    for each image i ∈ I do
7:        descriptions_i ← []
8:        
9:        for each prompt_set ps ∈ P do
10:           for epoch ← 1 to num_epochs do
11:               for each user_prompt up ∈ ps.user_prompts do
12:                   
13:                   ▷ Build conversation with optional system prompt
14:                   conversation ← BuildConversation(ps.system_prompt, up, i)
15:                   prompt ← processor.ApplyChatTemplate(conversation)
16:                   
17:                   ▷ Generate description with model
18:                   if ps.parse_json_output then
19:                       description ← GenerateWithJSONParsing(model, prompt, i, processor)
20:                   else
21:                       description ← GenerateDescription(model, prompt, i, processor)
22:                   end if
23:                   
24:                   descriptions_i.append(description)
25:               end for
26:           end for
27:        end for
28:        
29:        D[i] ← descriptions_i
30:    end for
31:    
32:    return D
33: end function

34: function GENERATEWITHJSONPARSING(model, prompt, image, processor)
35:    for attempt ← 1 to max_retries do
36:        output ← model.Generate(prompt, image)
37:        try
38:            ▷ Clean JSON formatting
39:            cleaned_output ← CleanJSONFormat(output)
40:            parsed_objects ← JSON.Parse(cleaned_output)
41:            return parsed_objects
42:        catch JSONError
43:            continue
44:        end try
45:    end for
46:    return ∅
47: end function

48: function GENERATEDESCRIPTION(model, prompt, image, processor)
49:    inputs ← processor.ProcessInputs(prompt, image)
50:    output ← model.Generate(inputs)
51:    description ← processor.DecodeOutput(output)
52:    return description
53: end function
```

### 2. Object Extraction and Cleaning

This stage extracts candidate objects from generated descriptions and applies a systematic three-step cleaning procedure to ensure high-quality candidates for verification.

```
Algorithm 3: Object Extraction and Cleaning
Require: Generated descriptions D, LLM extraction service
Ensure: Cleaned candidate objects C for each image

1: function OBJECTEXTRACTION(D)
2:    C ← ∅
3:    
4:    for each image i with descriptions D[i] do
5:        ▷ Step 1: Extract candidate objects from descriptions
6:        O ← ExtractCandidateObjects(D[i])
7:        
8:        ▷ Step 2: Apply cleaning procedure
9:        O' ← Clean(O)
10:       
11:       C[i] ← O'
12:    end for
13:    
14:    return C
15: end function

16: function EXTRACTCANDIDATEOBJECTS(descriptions)
17:    O ← set()
18:    for each description d ∈ descriptions do
19:        entities ← LLM.ExtractObjects(d)
20:        O.update(entities)
21:    end for
22:    return O
23: end function

24: function CLEAN(O)
25:    ▷ Step (a): Abstract concept removal
26:    O_physical ← ∅
27:    for each object o ∈ O do
28:        if IsPhysicalObject(o) then  ▷ Remove "atmosphere", "mood", etc.
29:            O_physical.add(o)
30:        end if
31:    end for
32:    
33:    ▷ Step (b): Noun singularization
34:    O_singular ← ∅
35:    for each object o ∈ O_physical do
36:        singular_form ← Singularize(o)  ▷ "dogs" → "dog"
37:        O_singular.add(singular_form)
38:    end for
39:    
40:    ▷ Step (c): Synonym merging
41:    O' ← ∅
42:    synonym_groups ← GroupSynonyms(O_singular)  ▷ "sofa" & "couch" → same group
43:    for each group g ∈ synonym_groups do
44:        representative ← SelectRepresentative(g)
45:        O'.add(representative)
46:    end for
47:    
48:    return O'
49: end function
```

### 3. Visual Verification

Using OWL-ViT v2 as the visual verifier, this stage employs a dual-threshold strategy to classify candidate objects as existing, hallucinated, or uncertain, prioritizing precision over recall.

```
Algorithm 4: Visual Verification
Require: Cleaned candidate objects O', images I, detector D, thresholds T_low, T_high
Ensure: Existing objects O_i, hallucinated objects O_h

1: function VISUALVERIFICATION(O', I, D)
2:    O_i ← ∅, O_h ← ∅
3:    
4:    for each image i with candidates O'[i] do
5:        if O'[i] = ∅ then
6:            continue  ▷ Skip if no candidates
7:        end if
8:        
9:        ▷ Expand query terms for better detection
10:       expanded_queries ← ExpandQueries(O'[i])
11:       
12:       ▷ Run object detection
13:       detection_scores ← D.Detect(i, expanded_queries)
14:       
15:       ▷ Map back to original candidates
16:       candidate_scores ← MapToOriginalCandidates(detection_scores, O'[i])
17:       
18:       ▷ Apply dual-threshold classification
19:       existing[i], hallucinated[i], uncertain[i] ← ClassifyByThreshold(candidate_scores)
20:       
21:       O_i[i] ← existing[i]
22:       O_h[i] ← hallucinated[i]
23:    end for
24:    
25:    return O_i, O_h
26: end function

27: function EXPANDQUERIES(candidates)
28:    expanded ← set()
29:    for each candidate c ∈ candidates do
30:        expanded.add(c.lower())
31:        plural_form ← Pluralize(c)  ▷ "dog" → "dogs"
32:        if plural_form ≠ c then
33:            expanded.add(plural_form.lower())
34:        end if
35:    end for
36:    return expanded
37: end function

38: function CLASSIFYBYTHRESHOLD(candidate_scores)
39:    existing, hallucinated, uncertain ← [], [], []
40:    
41:    for each candidate c with score s do
42:        if s > T_high then
43:            existing.append(c)
44:        else if s < T_low then
45:            hallucinated.append(c)
46:        else
47:            uncertain.append(c)  ▷ Discarded to ensure precision
48:        end if
49:    end for
50:    
51:    return existing, hallucinated, uncertain
52: end function
```

### 4. Question Construction

This stage generates two types of evaluation questions: basic existence questions for both existing and hallucinated objects, and AttriBait questions that test cognitive consistency by inquiring about attributes of non-existent objects.

```
Algorithm 5: Question Construction
Require: Existing objects O_i, hallucinated objects O_h, LLM question generator
Ensure: Basic questions Q_basic, AttriBait questions Q_attribait

1: function QUESTIONCONSTRUCTION(O_i, O_h)
2:    Q_basic ← ∅, Q_attribait ← ∅
3:    question_id ← 1
4:    
5:    ▷ Generate basic existence questions
6:    for each image i do
7:        ▷ Questions for existing objects (positive samples)
8:        for each object o ∈ O_i[i] do
9:            Q_basic.append({
10:               "id": question_id,
11:               "image": i,
12:               "text": "Is there a {o} in the image?",
13:               "label": "yes"
14:           })
15:           question_id ← question_id + 1
16:       end for
17:       
18:       ▷ Questions for hallucinated objects (negative samples)
19:       for each object o ∈ O_h[i] do
20:           Q_basic.append({
21:               "id": question_id,
22:               "image": i,
23:               "text": "Is there a {o} in the image?",
24:               "label": "no"
25:           })
26:           question_id ← question_id + 1
27:       end for
28:    end for
29:    
30:    ▷ Generate AttriBait questions for hallucinated objects
31:    for each image i do
32:        for each object o ∈ O_h[i] do
33:            attribait_q ← GenerateAttriBaitQuestion(o, i)
34:            if attribait_q ≠ ∅ then
35:                Q_attribait.append(attribait_q)
36:                question_id ← question_id + 1
37:            end if
38:        end for
39:    end for
40:    
41:    return Q_basic, Q_attribait
42: end function

43: function GENERATEATTRIBAITQUESTION(object, image)
44:    ▷ Generate attribute-focused multiple-choice question
45:    prompt ← BuildAttributePrompt(object)
46:    llm_response ← LLM.Generate(prompt)
47:    
48:    if ValidateResponse(llm_response) then
49:        options ← llm_response.options
50:        ▷ Add standard options
51:        options["C"] ← "Others"
52:        options["D"] ← "There is no {object} in the image"
53:        
54:        return {
55:            "id": question_id,
56:            "image": image,
57:            "text": llm_response.question,
58:            "options": options,
59:            "label": "D",  ▷ Correct answer
60:            "type": "attribait"
61:        }
62:    end if
63:    
64:    return ∅
65: end function
```

### 5. Model Evaluation

The final stage evaluates LVLMs on the constructed dataset, calculating comprehensive metrics including Hallucination Trigger Rate (HTR), accuracy, precision, and recall.

```
Algorithm 6: Model Evaluation
Require: Question dataset Q, images I, LVLM M, processor P
Ensure: Evaluation results R with statistics

1: function MODELEVALUATION(Q, I, M, P)
2:    R ← {"basic_results": [], "enhanced_results": []}
3:    
4:    ▷ Process basic existence questions
5:    for each question q ∈ Q.basic do
6:        result ← ProcessBasicQuestion(q, I, M, P)
7:        R.basic_results.append(result)
8:    end for
9:    
10:    ▷ Process AttriBait questions
11:    for each question q ∈ Q.enhanced do
12:        result ← ProcessEnhancedQuestion(q, I, M, P)
13:        R.enhanced_results.append(result)
14:    end for
15:    
16:    ▷ Calculate evaluation statistics
17:    statistics ← CalculateStatistics(R)
18:    R.statistics ← statistics
19:    
20:    return R
21: end function

22: function PROCESSBASICQUESTION(question, images, model, processor)
23:    image ← LoadImage(images[question.image])
24:    prompt ← BuildBasicPrompt(question.text)
25:    
26:    ▷ Generate model response
27:    response ← model.Generate(prompt, image)
28:    
29:    ▷ Parse yes/no answer
30:    predicted_answer ← ParseYesNoResponse(response)
31:    correct ← (predicted_answer = question.label)
32:    
33:    return {
34:        "question": question,
35:        "prediction": response,
36:        "predicted_answer": predicted_answer,
37:        "correct": correct
38:    }
39: end function

40: function PROCESSENHANCEDQUESTION(question, images, model, processor)
41:    image ← LoadImage(images[question.image])
42:    prompt ← BuildMultipleChoicePrompt(question.text, question.options)
43:    
44:    ▷ Generate model response
45:    response ← model.Generate(prompt, image)
46:    
47:    ▷ Parse A/B/C/D answer
48:    predicted_option ← ParseMultipleChoiceResponse(response)
49:    correct ← (predicted_option = question.label)
50:    
51:    return {
52:        "question": question,
53:        "prediction": response,
54:        "predicted_answer": predicted_option,
55:        "correct": correct
56:    }
57: end function

58: function CALCULATESTATISTICS(results)
59:    ▷ Calculate basic question metrics
60:    basic_stats ← CalculateBasicMetrics(results.basic_results)
61:    
62:    ▷ Calculate enhanced question metrics  
63:    enhanced_stats ← CalculateEnhancedMetrics(results.enhanced_results)
64:    
65:    return {
66:        "basic_questions": basic_stats,
67:        "enhanced_questions": enhanced_stats
68:    }
69: end function

70: function CALCULATEBASICMETRICS(basic_results)
71:    TP, FP, TN, FN ← 0, 0, 0, 0
72:    
73:    for each result r ∈ basic_results do
74:        if r.question.label = "yes" and r.predicted_answer = "yes" then
75:            TP ← TP + 1
76:        else if r.question.label = "no" and r.predicted_answer = "yes" then
77:            FP ← FP + 1  ▷ Hallucination triggered
78:        else if r.question.label = "no" and r.predicted_answer = "no" then
79:            TN ← TN + 1
80:        else if r.question.label = "yes" and r.predicted_answer = "no" then
81:            FN ← FN + 1
82:        end if
83:    end for
84:    
85:    HTR ← FP / (TN + FP)  ▷ Hallucination Trigger Rate
86:    accuracy ← (TP + TN) / (TP + FP + TN + FN)
87:    precision ← TP / (TP + FP)
88:    recall ← TP / (TP + FN)
89:    
90:    return {HTR, accuracy, precision, recall}
91: end function
```