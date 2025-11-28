def expert_generator_test(image2, question_type, question, options_text, domain_knowledge):

    # options_description = ', '.join([option.split(': ')[1] for option in options_text.split('\n')])

    descriptions_text = ""
    if isinstance(domain_knowledge, dict):
        descriptions_text = "\n".join([f"{key.capitalize()}: {value}" for key, value in domain_knowledge.items()])
    else:
        descriptions_text = domain_knowledge

    if not isinstance(question_type, str):
        raise ValueError(f"Question type must be a string, got {type(question_type).__name__}.")

    canonical_map = {
        "object frequency detection": "Object Frequency Detection",
        "traffic scene description": "Traffic Scene Description",
        "traffic analysis": "Traffic Analysis",
    }
    valid_types_display = "', '".join(canonical_map.values())
    normalized_key = question_type.strip().lower()
    if normalized_key not in canonical_map:
        raise ValueError(
            f"Invalid question type. Expected one of: '{valid_types_display}'. Received: '{question_type}'."
        )
    question_type = canonical_map[normalized_key]

    # Object Frequency Detection
    if question_type == "Object Frequency Detection":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image2},  # Only the query image
                    {
                        "type": "text",
                        "text": (
                            "Please analyze the provided traffic image to judge the level of congestion shown.\n"
                            "Domain Knowledge: Use the following information to support your reasoning:\n"
                            f"{descriptions_text}\n"
                            "Steps to follow:\n"
                            "1. Carefully observe the density and flow of vehicles, pedestrians, and traffic controls in the scene.\n"
                            "2. Evaluate indicators of congestion such as stop-and-go traffic, queues at intersections, or unused road capacity using the domain knowledge when helpful.\n"
                            "3. Decide which option best matches the congestion level depicted.\n"
                            f"Question: {question}\n"
                            f"Options:\n{options_text}\n"
                            "Respond with the letter of the correct option only (e.g., 'A' or 'B')."
                        )
                    },
                ],
            }
        ]
    # Traffic Scene Description
    elif question_type == "Traffic Scene Description":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image2},  # Only the query image
                    {
                        "type": "text",
                        "text": (
                            "The following domain knowledge summarises relevant traffic scene elements:\n"
                            f"Domain Knowledge:\n{descriptions_text}\n"
                            "Please analyze the provided image to answer the question.\n"
                            "Let's approach this systematically:\n"
                            "1. **Survey** the scene for key contextual cues such as road layout, traffic signals, weather conditions, and actor interactions.\n"
                            "2. **Identify** notable objects or events (e.g., vehicles stopped at lights, pedestrians crossing, maintenance work) leveraging the domain knowledge when necessary.\n"
                            "3. **Select** the option that best summarizes the overall traffic scene.\n"
                            f"Question: {question}\nOptions:\n{options_text}\nPlease respond with the letter of the correct option only."
                        )
                    },
                ],
            }
        ]
    # Traffic Analysis
    elif question_type == "Traffic Analysis":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image2},  # Only the query image
                    {
                        "type": "text",
                        "text": (
                            "Please analyze the provided image to answer the traffic-focused question based on observable evidence.\n"
                            "Let's approach this systematically:\n"
                            "1. **Inspect** the scene for traffic bottlenecks, incidents, rule violations, or other operational factors.\n"
                            "2. **Interpret** the potential causes or implications of these observations with reference to the domain knowledge.\n"
                            "3. **Conclude** which option best reflects the traffic condition or outcome described in the question.\n"
                            f"Question: {question}\nOptions:\n{options_text}\nPlease respond with the letter of the correct option only."
                        )
                    },
                ],
            }
        ]
    else:
        raise ValueError(f"Invalid question type. Expected one of: '{valid_types_display}'.")

    return messages