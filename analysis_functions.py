
def analyze_video_with_openai(video_url, video_metadata, language="English",api_key="",llm_name="Ollama",model=None):
    """Analyze video content with platform-specific optimization using OpenAI."""
    # Check if API key is available
    # if not os.environ.get("OPENAI_API_KEY"):
    #     raise Exception("OpenAI API key is required for analysis")
    

    # client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if llm_name=="OpenAI":
        if model==None:
            llm = ChatOpenAI(model="gpt-4o")
        else :
            llm = ChatOpenAI(model=model)

    elif llm_name=="Ollama":
        if model==None:
            llm=ChatOllama(model="llama3.2")
        else:
            llm=ChatOllama(model=model)


    elif llm_name=="Gemini":
        if model==None:
            llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        else:
            llm=ChatGoogleGenerativeAI(model=model)
    elif llm_name=="Groq":
        if model==None:
            llm=ChatGroq(model="llama-3.1-8b-instant")
        else:
            llm = ChatGroq(model=model)
    
    platform = video_metadata.get('platform', 'YouTube')

    analysis_prompt = f"""
    Analyze the {platform} video at {video_url} with title "{video_metadata.get('title', '')}".
    
    Provide a detailed analysis including:
    1. A summary of the video content (based on the title and any metadata)
    2. Main topics likely covered (at least 5 specific topics)
    3. Emotional tone and style of the video
    4. Target audience demographics and interests
    5. Content structure and flow
    
    Your analysis should be in {language} language.
    Make reasonable assumptions based on the available information.
    """
    analysis_message=[
            {"role": "system", "content": f"You are a video content analyst specialized in understanding {platform} video content, structure, and audience appeal. You are fluent in {language} and will provide all output in {language}."},
            {"role": "user", "content": analysis_prompt}
        ]

    # analysis_response = client.chat.completions.create(
    #     model="gpt-4",
    #     messages=[
    #         {"role": "system", "content": f"You are a video content analyst specialized in understanding {platform} video content, structure, and audience appeal. You are fluent in {language} and will provide all output in {language}."},
    #         {"role": "user", "content": analysis_prompt}
    #     ],
    #     temperature=0.7,
    # )
    # analysis_result = analysis_response.choices[0].message.content
    analysis_result=llm.invoke(analysis_message).content



    # Step 2: SEO Recommendations
    duration = video_metadata.get('duration', 0)
    minutes = duration // 60
    num_timestamps = min(15, max(5, int(minutes / 2))) if minutes > 0 else 5
    
    seo_prompt = f"""
    Based on this analysis of a {platform} video titled "{video_metadata.get('title', '')}":
    
    {analysis_result}
    
    Generate comprehensive SEO recommendations optimized specifically for {platform} including:
    
    1. EXACTLY 35 trending hashtags/tags related to the video content, ranked by potential traffic and relevance. 
       For {platform}, optimize the tags according to platform best practices.
    
    2. Detailed and SEO-optimized video description (400-500 words) that includes:
       - An engaging hook in the first 2-3 sentences that entices viewers
       - A clear value proposition explaining what viewers will gain
       - Key topics covered with strategic keyword placement
       - A strong call-to-action appropriate for {platform}
       - Essential links (placeholder)
       - Proper formatting with paragraph breaks for readability
    
    3. Exactly {num_timestamps} timestamps with descriptive labels evenly distributed throughout the video (duration: {duration} seconds)
    
    4. 5-7 alternative title suggestions ranked by SEO potential, each under 60 characters for YouTube or appropriate length for {platform}
    
    Format your response as JSON with the following structure:
    {{
        "tags": ["tag1", "tag2", ...],  // EXACTLY 35 tags total
        "description": "Complete optimized description here...",
        "timestamps": [
            {{"time": "00:00", "description": "Detailed segment description"}},
            ...
        ],
        "titles": [
            {{"rank": 1, "title": "Best title with keywords", "reason": "Why this title works well"}}
            ...
        ]
    }}
    
    All content should be in {language} language.
    Return ONLY the valid JSON with no explanation or other text.
    """
    seo_messages=[
            {"role": "system", "content": f"You are an SEO specialist focusing on optimizing {platform} content for maximum discovery and engagement. You are fluent in {language} and will provide all output in {language}."},
            {"role": "user", "content": seo_prompt}
        ]
    seo_result_text=llm.inovke(seo_messages).content
    
    # seo_response = client.chat.completions.create(
    #     model="gpt-4",
    #     messages=[
    #         {"role": "system", "content": f"You are an SEO specialist focusing on optimizing {platform} content for maximum discovery and engagement. You are fluent in {language} and will provide all output in {language}."},
    #         {"role": "user", "content": seo_prompt}
    #     ],
    #     temperature=0.7,
    # )
    
    # seo_result_text = seo_response.choices[0].message.content


    
    # Parse JSON from the SEO response
    try:
        seo_result = json.loads(seo_result_text)
        
        # Ensure we have exactly 35 tags
        if len(seo_result.get("tags", [])) != 35:
            seo_result["tags"] = ensure_exactly_35_tags(seo_result.get("tags", []), client, video_metadata, platform, language)
            
    except json.JSONDecodeError:
        # Try to extract JSON from the text if it's not pure JSON
        try:
            json_start = seo_result_text.find('{')
            json_end = seo_result_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                seo_result = json.loads(seo_result_text[json_start:json_end])
            else:
                # Generate a fallback JSON structure
                seo_result = generate_fallback_seo(video_metadata, platform, language)
        except:
            seo_result = generate_fallback_seo(video_metadata, platform, language)


# Step 3: Thumbnail Concepts
    thumbnail_prompt = f"""
    Based on this analysis of a {platform} video titled "{video_metadata.get('title', '')}":
    
    {analysis_result}
    
    Create 3 detailed thumbnail concepts specifically optimized for {platform}.
    
    For each concept, provide:
    1. The main visual elements to include (very specific and detailed)
    2. Text overlay suggestions (maximum 3-5 words, optimized for {platform})
    3. Color scheme (with exact hex codes for 3 colors)
    4. Focal point/main subject (detailed description of what should be the center of attention)
    5. Emotional tone the thumbnail should convey
    6. Composition details (layout, text placement, foreground/background elements)
    
    Format your response as JSON with the following structure:
    {{
        "thumbnail_concepts": [
            {{
                "concept": "Detailed description of concept 1",
                "text_overlay": "Short engaging text",
                "colors": ["#hexcode1", "#hexcode2", "#hexcode3"],
                "focal_point": "Specific description of focal element",
                "tone": "Emotional tone",
                "composition": "Layout and placement details"
            }},
            ...
        ]
    }}
    
    All content should be in {language} language.
    Return ONLY the valid JSON with no explanation.

    """
    
    thumbnail_message=[
            {"role": "system", "content": f"You are a professional thumbnail designer specialized in creating engaging {platform} thumbnails that maximize click-through rates. You understand the specific requirements and best practices for {platform} thumbnails. You are fluent in {language} and will provide all output in {language}."},
            {"role": "user", "content": thumbnail_prompt}
        ]
    thumbnail_result_text=llm.invoke(thumbnail_message)

    # thumbnail_response = client.chat.completions.create(
    #     model="gpt-4",
    #     messages=[
    #         {"role": "system", "content": f"You are a professional thumbnail designer specialized in creating engaging {platform} thumbnails that maximize click-through rates. You understand the specific requirements and best practices for {platform} thumbnails. You are fluent in {language} and will provide all output in {language}."},
    #         {"role": "user", "content": thumbnail_prompt}
    #     ],
    #     temperature=0.7,
    # )
    
    # thumbnail_result_text = thumbnail_response.choices[0].message.content

    # Parse JSON from the thumbnail response
    try:
        thumbnail_result = json.loads(thumbnail_result_text)
    except json.JSONDecodeError:
        # Try to extract JSON from the text if it's not pure JSON
        try:
            json_start = thumbnail_result_text.find('{')
            json_end = thumbnail_result_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                thumbnail_result = json.loads(thumbnail_result_text[json_start:json_end])
            else:
                thumbnail_result = generate_fallback_thumbnails(platform, language)
        except:
            thumbnail_result = generate_fallback_thumbnails(platform, language)
    
    return {
        "analysis": analysis_result,
        "seo": seo_result,
        "thumbnails": thumbnail_result
    }

