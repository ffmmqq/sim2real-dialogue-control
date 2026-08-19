"""
Compositional Agent Templates for HRL Museum Agent

Provides high-variety museum guide response generation without LLM calls.
Handles fact ID inclusion for ExplainNewFact and RepeatFact subactions.

Usage:
    from pipeline.utils.agent_templates import generate_agent_response_template
    
    response = generate_agent_response_template(
        option="Explain",
        subaction="ExplainNewFact",
        unmentioned_facts=["[DM_001] Created by Vermeer in 1665"],
        mentioned_facts=[],
        target_exhibit=None,
        rng=rng,
        visitor_state="ENGAGED"
    )

Toggle via: HRL_TEMPLATE_MODE=1

Critical: Output must include fact IDs in format [XX_NNN] for extraction.
Regex used: r'\\[([A-Z]{2}_\\d{3})\\]'

V3.0 - Massive variety expansion + responsiveness to visitor state
"""

import re
import random
from typing import List, Optional


# =============================================================================
# AGENT TEMPLATES BY SUBACTION - V3.0 Massive Variety + Responsiveness
# =============================================================================

AGENT_TEMPLATES = {
    # =========================================================================
    # EXPLAIN OPTION - Natural fact sharing like a real museum guide
    # Target: 50 intros × 20 outros = 1000+ combinations
    # =========================================================================
    
    "ExplainNewFact": {
        "intros": [
            # Enthusiastic sharing (for HIGHLY_ENGAGED visitors)
            "So here's what makes this really special - ",
            "You know what I find remarkable about this piece? ",
            "Here's something visitors often miss - ",
            "One of my favorite details about this is ",
            "What really stands out here is ",
            "Looking at this up close, you can see that ",
            "This is actually quite fascinating - ",
            "Here's a detail that always catches my attention: ",
            "Oh, this is wonderful - ",
            "I'm so excited to share this with you - ",
            "This is one of those details that just amazes me - ",
            "You're going to love this - ",
            "This is absolutely incredible - ",
            "I find this absolutely remarkable - ",
            "This detail is just stunning - ",
            "Here's something that will blow your mind - ",
            "This is one of my absolute favorites - ",
            "I'm thrilled to point this out - ",
            "This detail is extraordinary - ",
            "You'll find this fascinating - ",
            
            # Conversational sharing (for ENGAGED visitors)
            "So the thing about this piece is ",
            "What's interesting to note here is that ",
            "I always love pointing out that ",
            "Here's something that might surprise you - ",
            "You might not realize this, but ",
            "Take a look at this - ",
            "This might be my favorite part to share: ",
            "Here's an interesting detail - ",
            "Something worth noting is ",
            "I think you'll find this interesting - ",
            "Here's a neat detail - ",
            "This is worth mentioning - ",
            "I'd like to point out that ",
            "Here's something interesting - ",
            "This detail is noteworthy - ",
            "I think this is interesting - ",
            "Here's what's notable here - ",
            "This is something to notice - ",
            "I find this interesting - ",
            "Here's a detail worth noting - ",
            
            # Educational framing (for CURIOUS visitors)
            "An important thing to understand here is ",
            "To really appreciate this, you should know that ",
            "What gives this piece its significance is ",
            "The key detail here is ",
            "What makes this historically important is ",
            "To understand this fully, it helps to know that ",
            "The significance of this detail is ",
            "What's crucial to understand is ",
            "The important context here is ",
            "To appreciate this, consider that ",
            "The key to understanding this is ",
            "What's essential to know is ",
            "The critical detail is ",
            "To grasp the full picture, know that ",
            "The fundamental point is ",
            "What's key here is ",
            "The essential detail is ",
            "To fully understand, know that ",
            "The important point is ",
            "What's crucial here is ",
            
            # Simple, patient language (for CONFUSED/OVERLOADED visitors)
            "Simply put, ",
            "Basically, ",
            "In simple terms, ",
            "To put it simply, ",
            "The basic idea is ",
            "In plain language, ",
            "Simply stated, ",
            "At its core, ",
            "The simple version is ",
            "In straightforward terms, ",
        ],
        "outros": [
            "",
            " Pretty remarkable, right?",
            " It really shows the artistry.",
            " That's what makes it so special.",
            " Isn't that fascinating?",
            " I love that detail.",
            " It adds such depth to the piece.",
            " You can really see the skill involved.",
            " That's part of what makes this a masterpiece.",
            " Quite something when you think about it.",
            " Amazing, isn't it?",
            " Really impressive when you consider it.",
            " It's details like this that make art so special.",
            " I find that absolutely captivating.",
            " That's the kind of detail that stays with you.",
            " It really speaks to the artist's vision.",
            " That adds another layer of meaning.",
            " It's these details that tell the full story.",
            " That really enhances the whole piece.",
            " I think that's what makes this so memorable.",
        ],
    },
    
    "RepeatFact": {
        "intros": [
            # Helpful repetition
            "Let me put that another way - ",
            "So what I was saying is ",
            "To clarify, what that means is ",
            "In other words, ",
            "Just to make sure that's clear - ",
            "Let me rephrase that: ",
            "What I meant was ",
            "To put it simply, ",
            "Let me clarify - ",
            "To be clear, ",
            "What I'm saying is ",
            "To restate that - ",
            "Let me say that again - ",
            "To repeat - ",
            "Just to reiterate - ",
            
            # Patient re-explanation
            "So basically, the key point is ",
            "Right, so to recap - ",
            "The main thing to understand is ",
            "What's important to take away is ",
            "The essential point is ",
            "The key thing to remember is ",
            "The main detail is ",
            "What's crucial is ",
            "The important part is ",
            "The key element is ",
            "What matters here is ",
            "The significant detail is ",
            "The main point is ",
            "What's key is ",
            "The essential detail is ",
        ],
        "outros": [
            "",
            " Does that make more sense?",
            " Is that clearer now?",
            " Does that help?",
            " Let me know if you'd like me to explain further.",
            " Happy to go into more detail.",
            " Feel free to ask if anything's still unclear.",
            " Does that clarify things?",
            " Is that helpful?",
            " Let me know if you need more explanation.",
            " I'm here if you have questions.",
            " Does that answer your question?",
            " Is that what you were wondering about?",
            " Does that help you understand?",
            " Let me know if you'd like me to elaborate.",
        ],
    },
    
    "ClarifyFact": {
        "intros": [
            # Simplification
            "Think of it this way - ",
            "A good way to think about it is ",
            "Basically what that means is ",
            "In everyday terms, ",
            "The simple version is ",
            "Imagine it like this: ",
            "Picture it this way - ",
            "To simplify - ",
            "In plain terms, ",
            "The easiest way to understand it is ",
            "Think about it like this - ",
            "A simple way to see it is ",
            "To make it simple - ",
            "In basic terms, ",
            "The straightforward version is ",
            
            # Analogy-based
            "It's similar to how ",
            "You could compare it to ",
            "It's kind of like when ",
            "Picture it as ",
            "Think of it as being like ",
            "It's comparable to ",
            "You might think of it as ",
            "It's analogous to ",
            "Imagine it as ",
            "It's like when ",
        ],
        "outros": [
            "",
            " Does that analogy help?",
            " Make sense?",
            " Is that a helpful way to think about it?",
            " Hopefully that's clearer.",
            " That's the gist of it.",
            " Does that help clarify?",
            " Is that easier to understand?",
            " Does that make it clearer?",
            " Is that a better way to see it?",
            " Does that help you picture it?",
            " Is that more understandable?",
        ],
    },
    
    # =========================================================================
    # ASK QUESTION OPTION - Engaging, conversational questions
    # Target: 50+ templates per subaction
    # =========================================================================
    
    "AskOpinion": {
        "templates": [
            # Inviting personal reactions
            "I'm curious - what's your first impression when you look at this?",
            "What catches your eye most about this piece?",
            "How does this make you feel when you look at it?",
            "What do you notice first about this painting?",
            "Does this evoke any particular mood or feeling for you?",
            "What's your initial reaction to this piece?",
            "What strikes you when you first see this?",
            "What's your gut feeling about this artwork?",
            "What emotions does this bring up for you?",
            "What's your immediate response to this?",
            
            # Encouraging engagement
            "What draws your attention here?",
            "I'd love to hear your thoughts - what stands out to you?",
            "Looking at this, what comes to mind?",
            "What's your gut reaction to this piece?",
            "If you had to describe this to a friend, what would you say?",
            "I'm really interested in your perspective - what do you think?",
            "What's your take on this piece?",
            "I'd be curious to know what you're thinking.",
            "What's running through your mind as you look at this?",
            "I'd love to hear what you're noticing.",
            "What's your perspective on this?",
            "I'm interested in what you see.",
            "What are your thoughts?",
            "I'd like to hear your view.",
            "What's your opinion?",
            
            # Specific elements
            "What do you think of the colors the artist chose here?",
            "Does the composition strike you in any particular way?",
            "What do you make of the subject's expression?",
            "Is there anything here that surprises you?",
            "What details are you drawn to?",
            "What do you think about the technique used here?",
            "How do you feel about the overall mood of this piece?",
            "What's your reaction to the level of detail?",
            "What do you notice about the lighting?",
            "How do you respond to the composition?",
            "What's your take on the artist's style?",
            "What do you think about the subject matter?",
            "How do you feel about the color palette?",
            "What's your impression of the brushwork?",
            "What do you notice about the perspective?",
            "How do you respond to the texture?",
            "What's your reaction to the overall effect?",
            "What do you think about the storytelling here?",
            "How do you feel about the emotional impact?",
            "What's your take on the historical context?",
        ],
    },
    
    "AskMemory": {
        "templates": [
            # Gentle recall questions
            "Do you remember what we mentioned about how this was made?",
            "Can you recall the technique we talked about?",
            "What did we say about when this was created?",
            "Remember the story behind this commission?",
            "Do you recall what made this artist famous?",
            "Can you remember the date we discussed?",
            "Do you recall the historical context we covered?",
            "Remember what we said about the symbolism?",
            "Can you recall the artist's name?",
            "Do you remember the period we talked about?",
            
            # Connecting to earlier discussion
            "Going back to what we discussed - do you remember the key detail?",
            "We touched on the historical context - does any of that stick with you?",
            "Can you remember what we said about the symbolism here?",
            "What detail do you remember most from our discussion?",
            "Does any of what we covered stand out in your memory?",
            "Thinking back to our conversation - what comes to mind?",
            "From what we've talked about, what do you recall?",
            "Looking back, what detail stands out to you?",
            "Of everything we've discussed, what do you remember?",
            "What from our conversation has stuck with you?",
            "Can you recall any of the details we covered?",
            "What do you remember from what we've discussed?",
            "Does anything from our talk come back to you?",
            "What details have you retained?",
            "What's stayed with you from our discussion?",
            "Can you recall the main points?",
            "What do you remember?",
            "Does anything come to mind?",
            "What sticks out?",
            "What do you recall?",
        ],
    },
    
    "AskClarification": {
        "templates": [
            # Understanding visitor interest
            "What aspect are you most curious about - the technique or the history?",
            "Would you like me to focus more on the artistic side or the backstory?",
            "What would be most interesting for you to learn about this?",
            "Is there something specific you'd like me to go deeper on?",
            "Would more context about the artist help, or about the painting itself?",
            "What would you like to explore further?",
            "What area interests you most?",
            "What would you like to understand better?",
            "What aspect would you like me to explain?",
            "What would be most helpful for you?",
            
            # Gauging understanding
            "What would be most helpful for me to explain?",
            "Is there anything you'd like me to clarify or expand on?",
            "What questions are coming to mind as you look at this?",
            "Any particular element you want to understand better?",
            "What part of this interests you most?",
            "What would you like to know more about?",
            "What should I focus on explaining?",
            "What would help you understand this better?",
            "What aspect needs more explanation?",
            "What would clarify things for you?",
            "What do you want to know?",
            "What interests you?",
            "What would help?",
            "What should I explain?",
            "What do you need?",
            "What would be useful?",
            "What should I focus on?",
            "What would clarify?",
            "What interests you most?",
            "What would help you?",
        ],
    },
    
    # =========================================================================
    # OFFER TRANSITION OPTION - Smooth, natural transitions
    # Target: 40+ templates per subaction
    # =========================================================================
    
    "SuggestMove": {
        "templates": [
            # Enthusiastic suggestions
            "Speaking of which, there's a piece just over there that I think you'd really enjoy - {target}. Shall we?",
            "You know what would be a great follow-up to this? {target} is right around the corner.",
            "I think you'd love what's next - {target} has some fascinating connections to what we've been discussing.",
            "Ready to see something special? {target} is one of my favorites to show people.",
            "I'm excited to show you {target} next - it's really something special.",
            "You're going to love {target} - it's just over there.",
            "I think {target} would be perfect for you - shall we head over?",
            "Wait until you see {target} - it's remarkable.",
            "I'd love to show you {target} - it's one of my favorites.",
            "You have to see {target} - it's incredible.",
            
            # Natural transitions
            "Have you seen {target} yet? I think you'd find it really interesting.",
            "If you liked this, wait until you see {target}. It's just over here.",
            "There's another piece nearby - {target} - that I think you'd appreciate.",
            "Shall we head over to {target}? It pairs beautifully with what we just saw.",
            "I think you'd enjoy {target} - it's right through here.",
            "There's something special about {target} - want to see it?",
            "I think {target} would interest you - shall we?",
            "You might like {target} - it's nearby.",
            "I think {target} is worth seeing - ready?",
            "There's {target} over there - interested?",
            
            # Gentle suggestions
            "When you're ready, {target} is a wonderful next stop.",
            "I'd love to show you {target} next if you're interested.",
            "Let me know when you'd like to move on - {target} is right through here.",
            "{target} is worth seeing while we're in this area.",
            "If you're ready, {target} is waiting.",
            "When you want to continue, {target} is there.",
            "If you'd like, we can see {target} next.",
            "Whenever you're ready, {target} is available.",
            "If you want, {target} is nearby.",
            "When ready, {target} awaits.",
            
            # Contextual transitions
            "This connects beautifully to {target} - shall we see it?",
            "Speaking of which, {target} relates to this - want to check it out?",
            "This leads nicely to {target} - interested?",
            "Following from this, {target} makes sense - shall we?",
            "Building on this, {target} is next - ready?",
        ],
    },
    
    "SummarizeAndSuggest": {
        "templates": [
            # Wrapping up with transition
            "We've covered some wonderful ground with this piece. Ready to discover {target}? I think you'll love it.",
            "That gives you a good sense of this masterpiece. Shall we continue on to {target}?",
            "What a fascinating work! Speaking of which, {target} has an interesting connection to what we just discussed.",
            "We've really explored this one thoroughly. {target} would be a great next step.",
            "I think we've done this piece justice. {target} is waiting for us.",
            "This has been a wonderful discussion. {target} would complement it nicely.",
            "We've learned a lot about this one. {target} is next on our journey.",
            "This piece has so much to offer. {target} has its own story to tell.",
            "We've really delved into this. {target} would be perfect to see next.",
            "I think we've covered this well. {target} is ready when you are.",
            
            # Appreciative transitions
            "This has been a lovely discussion! {target} is waiting for us just over there.",
            "I always enjoy sharing this piece. Now, let me show you {target}.",
            "We've really done this painting justice. {target} would be a perfect next stop.",
            "This has been wonderful. {target} is our next destination.",
            "I've loved sharing this with you. {target} is next.",
            "This has been great. {target} awaits.",
            "Wonderful discussion. {target} is next.",
            "Great conversation. {target} is ready.",
            "Lovely talk. {target} is there.",
            "Perfect. {target} is next.",
            
            # Conversational wrap-up
            "Any last thoughts before we head to {target}? It's one you won't want to miss.",
            "Take one more look if you'd like, and then let's explore {target}.",
            "Now that we've appreciated this one, shall we see what {target} has to offer?",
            "If you're satisfied with this one, {target} is our next stop.",
            "When you're ready, {target} is there for us.",
            "If you've seen enough here, {target} is next.",
            "Once you're ready, {target} awaits.",
            "When ready, {target} is there.",
            "If ready, {target} is next.",
            "When you are, {target} is waiting.",
        ],
    },
    
    # =========================================================================
    # CONCLUDE OPTION - Warm, memorable endings
    # Target: 40+ templates
    # =========================================================================
    
    "WrapUp": {
        "templates": [
            # Warm closings
            "What a wonderful journey through the collection! Thank you for being such an engaged visitor.",
            "It's been my pleasure sharing these masterpieces with you. I hope they leave a lasting impression.",
            "Thank you for your curiosity and great questions! I hope you'll come back and explore more.",
            "This has been such a lovely tour. Art is always more enjoyable when shared with interested visitors.",
            "What an amazing experience we've had together! Thank you for your wonderful engagement.",
            "This has been such a pleasure. Thank you for being such a thoughtful visitor.",
            "I've really enjoyed our time together. Thank you for your interest and questions.",
            "What a delightful tour! Thank you for making it so enjoyable.",
            "This has been wonderful. Thank you for being such an engaged participant.",
            "I've loved sharing these pieces with you. Thank you for your enthusiasm.",
            
            # Appreciative endings
            "I've really enjoyed our conversation about these works. Thank you for your time!",
            "It was wonderful exploring these pieces together. Take care, and enjoy the rest of your visit!",
            "Thank you for letting me share some of my favorite artworks with you!",
            "I hope these paintings have given you something to think about. Enjoy the rest of the museum!",
            "Thank you for such a wonderful visit. I hope you enjoyed it as much as I did.",
            "It's been a pleasure. Thank you for your time and interest.",
            "I've enjoyed this immensely. Thank you for being here.",
            "This has been great. Thank you for visiting.",
            "Thank you for a lovely tour. Enjoy the rest of your day.",
            "I've loved this. Thank you so much.",
            
            # Memorable closings
            "What a great tour we've had! I hope the stories behind these pieces stay with you.",
            "Thank you for being such a thoughtful visitor. Art needs people who really look and listen.",
            "I hope these pieces have inspired you. Thank you for your wonderful engagement.",
            "What a memorable experience! Thank you for making it special.",
            "I hope these artworks stay with you. Thank you for your thoughtful attention.",
            "Thank you for such a meaningful visit. I hope you'll remember these pieces.",
            "I hope this tour has been meaningful. Thank you for your engagement.",
            "Thank you for making this special. I hope you'll carry these pieces with you.",
            "I hope these works have touched you. Thank you for your presence.",
            "Thank you for a memorable tour. I hope these pieces resonate with you.",
            "I hope you've enjoyed this. Thank you for your time.",
            "Thank you for visiting. I hope it was meaningful.",
            "I hope you enjoyed it. Thank you.",
            "Thank you for coming. I hope it was special.",
            "I hope it was good. Thank you.",
            "Thank you. I hope you enjoyed.",
            "I hope it was meaningful.",
            "Thank you for visiting.",
            "I hope you enjoyed.",
            "Thank you.",
        ],
    },
    
    "SummarizeKeyPoints": {
        "intros": [
            "What a journey we've had today! We explored ",
            "Looking back on our tour, we covered ",
            "We've seen some remarkable pieces today - ",
            "Let me recap our wonderful tour: we discovered ",
            "So today we've experienced ",
            "To summarize our journey, we've explored ",
            "Looking back, we've seen ",
            "Our tour today included ",
            "We've discovered ",
            "Together we've explored ",
        ],
        "outros": [
            " and learned about the incredible stories behind them.",
            " - each with its own unique history and significance.",
            " and so much about the Dutch Golden Age.",
            ". What a rich collection this is!",
            " and the artists who created these masterpieces.",
            ". I hope these pieces stay with you.",
            " and their fascinating backgrounds.",
            " - what an amazing collection!",
            " and the rich history they represent.",
            ". I hope this tour has been meaningful.",
            " and learned so much together.",
            ". Thank you for exploring with me.",
            " and discovered their stories.",
            ". I hope you'll remember this.",
            " and shared wonderful moments.",
        ],
    },
}


# =============================================================================
# RESPONSIVENESS LOGIC - Tone Selection Based on Visitor State
# =============================================================================

def _select_agent_tone(
    visitor_state: Optional[str],
    intros: List[str],
    outros: List[str],
    rng: random.Random
) -> tuple:
    """
    Select intro/outro based on visitor state for responsiveness.
    
    Returns:
        Tuple of (intro, outro) selected based on visitor state
    """
    if not visitor_state:
        # No visitor state info - random selection
        return rng.choice(intros), rng.choice(outros)
    
    visitor_state = visitor_state.upper()
    
    if visitor_state == "CONFUSED" or visitor_state == "OVERLOADED":
        # Prefer simpler, patient intros
        simple_keywords = ["simply", "basically", "think of", "plain", "simple", "straightforward", "easy"]
        simple_intros = [i for i in intros if any(word in i.lower() for word in simple_keywords)]
        if simple_intros:
            intro = rng.choice(simple_intros)
        else:
            intro = rng.choice(intros)
        # Simpler outros too
        simple_outros = [o for o in outros if not any(word in o.lower() for word in ["fascinating", "remarkable", "amazing"])]
        if simple_outros:
            outro = rng.choice(simple_outros)
        else:
            outro = rng.choice(outros)
    
    elif visitor_state == "HIGHLY_ENGAGED":
        # Prefer enthusiastic intros
        enthusiastic_keywords = ["amazing", "fascinating", "incredible", "remarkable", "wonderful", "special", "love", "excited"]
        enthusiastic_intros = [i for i in intros if any(word in i.lower() for word in enthusiastic_keywords)]
        if enthusiastic_intros:
            intro = rng.choice(enthusiastic_intros)
        else:
            intro = rng.choice(intros)
        # More enthusiastic outros
        enthusiastic_outros = [o for o in outros if any(word in o.lower() for word in ["fascinating", "remarkable", "amazing", "incredible"])]
        if enthusiastic_outros:
            outro = rng.choice(enthusiastic_outros)
        else:
            outro = rng.choice(outros)
    
    elif visitor_state == "CURIOUS":
        # Prefer educational, detailed intros
        educational_keywords = ["understand", "appreciate", "significance", "important", "key", "crucial", "essential", "context"]
        educational_intros = [i for i in intros if any(word in i.lower() for word in educational_keywords)]
        if educational_intros:
            intro = rng.choice(educational_intros)
        else:
            intro = rng.choice(intros)
        outro = rng.choice(outros)
    
    elif visitor_state == "FATIGUED" or visitor_state == "DISENGAGED":
        # Prefer brief, minimal intros
        brief_intros = [i for i in intros if len(i.split()) <= 8]  # Shorter intros
        if brief_intros:
            intro = rng.choice(brief_intros)
        else:
            intro = rng.choice(intros)
        # Minimal outros
        minimal_outros = [o for o in outros if len(o) <= 15 or o == ""]
        if minimal_outros:
            outro = rng.choice(minimal_outros)
        else:
            outro = rng.choice(outros)
    
    elif visitor_state == "BORED_OF_TOPIC":
        # Prefer dynamic, transition-focused language (but this is for Explain, so just neutral)
        intro = rng.choice(intros)
        outro = rng.choice(outros)
    
    else:
        # ENGAGED or other - balanced selection
        intro = rng.choice(intros)
        outro = rng.choice(outros)
    
    return intro, outro


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_fact_id(fact_with_id: str) -> str:
    """
    Extract fact ID from '[XX_NNN] content' format.
    
    Args:
        fact_with_id: Full fact string like "[DM_001] The artwork was created..."
    
    Returns:
        Fact ID like "DM_001" or empty string if not found
    """
    if fact_with_id.startswith('[') and ']' in fact_with_id:
        return fact_with_id[1:fact_with_id.index(']')]
    return ""


def strip_fact_id(fact_with_id: str) -> str:
    """
    Remove fact ID, return just content.
    
    Args:
        fact_with_id: Full fact string like "[DM_001] The artwork was created..."
    
    Returns:
        Content without ID like "The artwork was created..."
    """
    if fact_with_id.startswith('[') and ']' in fact_with_id:
        return fact_with_id[fact_with_id.index(']')+1:].strip()
    return fact_with_id


def format_exhibit_name(exhibit_id: str) -> str:
    """
    Format exhibit ID for natural language use.
    
    Args:
        exhibit_id: Raw exhibit ID like "Delft_Masterpiece"
    
    Returns:
        Formatted name like "the Delft Masterpiece"
    """
    if not exhibit_id:
        return "the next piece"
    name = exhibit_id.replace("_", " ")
    # Add article for naturalness
    if not name.lower().startswith("the "):
        name = f"the {name}"
    return name


def make_fact_conversational(fact_content: str) -> str:
    """
    Make a fact string flow more naturally in conversation.
    Handles cases where the fact might be dry metadata.
    
    Args:
        fact_content: Raw fact content like "Style: Oil on panel"
    
    Returns:
        More conversational version like "this was painted in oil on panel"
    """
    content = fact_content.strip()
    
    # Handle "Key: Value" format metadata
    if ':' in content and len(content.split(':')) == 2:
        key, value = content.split(':', 1)
        key = key.strip().lower()
        value = value.strip()
        
        # Transform common metadata patterns
        if key in ('style', 'medium', 'technique'):
            return f"this was created using {value.lower()}"
        elif key in ('date', 'year', 'period'):
            return f"this dates from {value}"
        elif key in ('artist', 'painter', 'creator'):
            return f"this was painted by {value}"
        elif key in ('size', 'dimensions'):
            return f"it measures {value}"
        elif key in ('location', 'origin'):
            return f"it comes from {value}"
        else:
            return f"the {key} is {value.lower()}"
    
    # Make sure it starts lowercase if being integrated into a sentence
    if content and content[0].isupper():
        # Check if it's not a proper noun (rough heuristic)
        first_word = content.split()[0] if content.split() else ""
        if first_word.lower() in ('the', 'this', 'it', 'a', 'an'):
            content = content[0].lower() + content[1:]
    
    return content


# =============================================================================
# MAIN GENERATION FUNCTION
# =============================================================================

def generate_agent_response_template(
    option: str,
    subaction: str,
    unmentioned_facts: List[str],
    mentioned_facts: List[str],
    target_exhibit: Optional[str],
    rng: random.Random,
    visitor_state: Optional[str] = None,
    last_visitor_utterance: Optional[str] = None
) -> str:
    """
    Generate agent response with proper fact IDs without LLM.
    
    Args:
        option: High-level option (Explain, AskQuestion, OfferTransition, Conclude)
        subaction: Specific subaction (ExplainNewFact, RepeatFact, etc.)
        unmentioned_facts: List of facts not yet mentioned (format: "[XX_NNN] content")
        mentioned_facts: List of facts already mentioned (format: "[XX_NNN] content")
        target_exhibit: For transitions, the exhibit to suggest
        rng: Seeded random generator for reproducibility
        visitor_state: Visitor's current state (for responsiveness)
        last_visitor_utterance: Visitor's last utterance (for future use)
    
    Returns:
        Generated agent response string with fact IDs where appropriate
    """
    templates = AGENT_TEMPLATES.get(subaction, {})
    
    # =========================================================================
    # EXPLAIN OPTION
    # =========================================================================
    
    if subaction == "ExplainNewFact":
        if not unmentioned_facts:
            fallbacks = [
                "I think we've covered all the key details about this piece. Shall we explore something new?",
                "That's pretty much everything I wanted to share about this one. Ready to move on?",
                "We've really explored this piece thoroughly! Want to see what else is here?",
            ]
            return rng.choice(fallbacks)
        
        # DETERMINISTIC: Pick first unmentioned fact
        fact_full = unmentioned_facts[0]
        fact_id = extract_fact_id(fact_full)
        fact_content = strip_fact_id(fact_full)
        
        # Make the fact content more conversational
        fact_content = make_fact_conversational(fact_content)
        
        # Use responsiveness to select tone
        intro, outro = _select_agent_tone(visitor_state, templates["intros"], templates["outros"], rng)
        
        # CRITICAL: Include fact ID in brackets for extraction
        return f"{intro}{fact_content} [{fact_id}]{outro}"
    
    elif subaction == "RepeatFact":
        if not mentioned_facts:
            return "Let me share something about this piece with you."
        
        # Use last mentioned fact
        fact_full = mentioned_facts[-1]
        fact_id = extract_fact_id(fact_full)
        fact_content = strip_fact_id(fact_full)
        
        # Make the fact content more conversational
        fact_content = make_fact_conversational(fact_content)
        
        # Use responsiveness to select tone
        intro, outro = _select_agent_tone(visitor_state, templates["intros"], templates["outros"], rng)
        
        # CRITICAL: Include fact ID in brackets for extraction
        return f"{intro}{fact_content} [{fact_id}]{outro}"
    
    elif subaction == "ClarifyFact":
        if not mentioned_facts:
            fallbacks = [
                "Let me try explaining that differently.",
                "Let me put that in simpler terms.",
                "Here's another way to think about it.",
            ]
            return rng.choice(fallbacks)
        
        fact_content = strip_fact_id(mentioned_facts[-1])
        fact_content = make_fact_conversational(fact_content)
        
        # Use responsiveness to select tone
        intro, outro = _select_agent_tone(visitor_state, templates["intros"], templates["outros"], rng)
        
        # NO fact ID for clarification (per original design)
        return f"{intro}{fact_content}{outro}"
    
    # =========================================================================
    # ASK QUESTION OPTION
    # =========================================================================
    
    elif subaction in ("AskOpinion", "AskMemory", "AskClarification"):
        # For questions, responsiveness could adjust question style, but for now just random
        return rng.choice(templates["templates"])
    
    # =========================================================================
    # OFFER TRANSITION OPTION
    # =========================================================================
    
    elif subaction in ("SuggestMove", "SummarizeAndSuggest"):
        target_name = format_exhibit_name(target_exhibit)
        template = rng.choice(templates["templates"])
        return template.format(target=target_name)
    
    # =========================================================================
    # CONCLUDE OPTION
    # =========================================================================
    
    elif subaction == "WrapUp":
        return rng.choice(templates["templates"])
    
    elif subaction == "SummarizeKeyPoints":
        intro = rng.choice(templates["intros"])
        outro = rng.choice(templates["outros"])
        return f"{intro}some wonderful artworks from the collection{outro}"
    
    # =========================================================================
    # FALLBACK
    # =========================================================================
    
    fallbacks = [
        "This is a truly remarkable piece. What do you think?",
        "There's so much to appreciate here.",
        "I love sharing this piece with visitors.",
    ]
    return rng.choice(fallbacks)


# =============================================================================
# TESTING / DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    # Quick demonstration
    rng = random.Random(42)
    
    print("=" * 70)
    print("AGENT TEMPLATE V3.0 DEMONSTRATION")
    print("=" * 70)
    
    # Test facts - including dry metadata style
    test_unmentioned = [
        "[DM_001] The artwork was created by Johannes Vermeer in 1665",
        "[DM_002] Uses rare ultramarine blue pigment",
        "[DM_003] Style: Oil on canvas",  # Metadata format
        "[DM_004] Size: 46.5 x 39 cm",  # Metadata format
        "[DM_005] The painting depicts a woman reading a letter by a window",
    ]
    test_mentioned = [
        "[DM_001] The artwork was created by Johannes Vermeer in 1665",
    ]
    
    print("\n--- ExplainNewFact (with responsiveness) ---")
    for state in ["HIGHLY_ENGAGED", "CONFUSED", "ENGAGED", None]:
        print(f"\nVisitor State: {state}")
        resp = generate_agent_response_template(
            "Explain", "ExplainNewFact", test_unmentioned, [], None, rng, visitor_state=state
        )
        print(f"  {resp}")
        match = re.search(r'\[([A-Z]{2}_\d{3})\]', resp)
        print(f"  Fact ID: {match.group(1) if match else 'NONE'}")
    
    print("\n--- RepeatFact ---")
    for _ in range(3):
        resp = generate_agent_response_template(
            "Explain", "RepeatFact", test_unmentioned, test_mentioned, None, rng
        )
        print(f"  {resp}")
    
    print("\n--- AskOpinion ---")
    for _ in range(5):
        resp = generate_agent_response_template(
            "AskQuestion", "AskOpinion", [], [], None, rng
        )
        print(f"  {resp}")
    
    print("\n--- SuggestMove ---")
    for _ in range(4):
        resp = generate_agent_response_template(
            "OfferTransition", "SuggestMove", [], [], "Turkish_Portrait", rng
        )
        print(f"  {resp}")
    
    print("\n--- SummarizeAndSuggest ---")
    for _ in range(3):
        resp = generate_agent_response_template(
            "OfferTransition", "SummarizeAndSuggest", [], [], "Kitchen_Scene", rng
        )
        print(f"  {resp}")
    
    print("\n--- WrapUp ---")
    for _ in range(3):
        resp = generate_agent_response_template(
            "Conclude", "WrapUp", [], [], None, rng
        )
        print(f"  {resp}")
    
    print("\n" + "=" * 70)
    print("TEMPLATE STATISTICS")
    print("=" * 70)
    for subaction, content in AGENT_TEMPLATES.items():
        if "templates" in content:
            count = len(content["templates"])
        else:
            count = len(content.get("intros", [])) * len(content.get("outros", []))
        print(f"  {subaction}: ~{count} combinations")
    print("=" * 70)
