def expert_generator_test(image2, question_type, question, options_text, domain_knowledge):

    # options_description = ', '.join([option.split(': ')[1] for option in options_text.split('\n')])

    descriptions_text = ""
    if isinstance(domain_knowledge, dict):
        descriptions_text = "\n".join([f"{key.capitalize()}: {value}" for key, value in domain_knowledge.items()])
    else:
        descriptions_text = domain_knowledge

    # Gray Industry Detection
    if question_type == "Gray Industry Detection":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image2},  # Only the query image
                    {
                        "type": "text",
                        "text": (
                            "Please analyze the provided image to determine if it contains elements that promote gray industry activities.\n"
                            "Domain Knowledge: Use the following information to assist your analysis:\n"
                            f"{descriptions_text}\n"
                            "Steps to follow:\n"
                            "1. Carefully examine the image for any suspicious elements, such as unusual text, design, or content.\n"
                            "2. Compare the observed elements with the provided domain knowledge.\n"
                            "3. Decide whether the image contains elements promoting gray industry activities.\n"
                            "Question: {question}\n"
                            "Options:\n{options_text}\n"
                            "Respond with the letter of the correct option only (e.g., 'A' or 'B')."
                        )
                    },
                ],
            }
        ]
    # Gray Industry Description
    elif question_type == "Gray Industry Description":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image2},  # Only the query image
                    {
                        "type": "text",
                        "text": (
                            "Following is the domain knowledge which contains descriptions of potential grey industry characteristics:\n"
                            f"Domain Knowledge:\n{descriptions_text}\n"
                            "Please analyze the provided image to answer the question.\n"
                            "Let's approach this systematically:\n"
                            "1. **Analyze** the image for key elements like text, user interface design, and requested permissions.\n"
                            "2. **Identify** any features that suggest it might be a grey industry application, based on the domain knowledge or general understanding.\n"
                            "3. **Decide** which option best describes the characteristics of the application shown in the image.\n"
                            f"Question: {question}\nOptions:\n{options_text}\nPlease respond with the letter of the correct option only."
                        )
                    },
                ],
            }
        ]
    # Gray Industry Analysis
    elif question_type == "Gray Industry Analysis":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image2},  # Only the query image
                    {
                        "type": "text",
                        "text": (
                            "Please analyze the provided image to answer the question based on its content.\n"
                            "Let's approach this systematically:\n"
                            "1. **Examine** the image for any suspicious elements, such as unusual permissions or misleading UI design.\n"
                            "2. **Evaluate** the potential risks associated with these elements, considering user privacy and data security.\n"
                            "3. **Conclude** which option best describes the risks or characteristics of the application shown in the image.\n"
                            f"Question: {question}\nOptions:\n{options_text}\nPlease respond with the letter of the correct option only."
                        )
                    },
                ],
            }
        ]
    else:
        raise ValueError("Invalid question type. Expected one of: 'Gray Industry Detection', 'Gray Industry Description', 'Gray Industry Analysis'.")

    return messages