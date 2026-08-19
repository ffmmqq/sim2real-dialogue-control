"""
Compositional Visitor Templates for HRL Museum Agent

Provides high-variety visitor utterance generation without LLM calls.
Works with both Sim8 (response_type) and StateMachine (visitor_state) simulators.

Usage:
    from simulator.visitor_templates import generate_visitor_utterance
    
    # For Sim8 (uses response_type)
    utterance = generate_visitor_utterance(response_type="question", visitor_state=None, aoi="painting", rng=rng, agent_option="Explain", agent_subaction="ExplainNewFact")
    
    # For StateMachine (uses visitor_state)
    utterance = generate_visitor_utterance(response_type=None, visitor_state="ENGAGED", aoi="painting", rng=rng, agent_option="Explain", agent_subaction="ExplainNewFact")

Toggle via: HRL_TEMPLATE_MODE=1

V3.0 - Massive variety expansion + responsiveness to agent actions
"""

import re
import random
from typing import Optional, Dict, List


# =============================================================================
# VISITOR RESPONSE TEMPLATES - ORGANIZED BY CATEGORY FOR RESPONSIVENESS
# =============================================================================

VISITOR_RESPONSES = {
    # =========================================================================
    # HIGHLY_ENGAGED - Excited visitors asking follow-ups, sharing reactions
    # Target: 80+ templates total
    # =========================================================================
    
    "HIGHLY_ENGAGED": {
        "general": [
            # Excited reactions with follow-up questions
            "Oh wow, that's incredible! I had no idea. How did they even figure that out back then?",
            "Wait, really? That's fascinating! Is that why the colors look so vivid here?",
            "No way! That's so cool. Were there other artists doing similar things at the time?",
            "That's amazing! I love how you can see the brushwork up close. What technique is that called?",
            "Oh I love that! The detail is stunning. Did it take them a long time to paint this?",
            "Wow, I never would have noticed that! What else should I be looking for here?",
            "That's beautiful! I can see why this is considered a masterpiece. What makes it so special compared to others?",
            "Oh that's so interesting! I wonder what the artist was thinking when they made that choice.",
            "That's incredible detail! How did artists learn these techniques back then?",
            "Fascinating! This really changes how I see it. Can you tell me more about the symbolism?",
            "Oh my gosh, that's remarkable! I had no idea about that connection.",
            "This is absolutely stunning! How did they manage to capture that level of detail?",
            "Wow, I'm blown away! This is so much more complex than I realized.",
            "That's extraordinary! I can't believe I never noticed that before.",
            "Oh this is incredible! The craftsmanship here is just mind-blowing.",
            "Wait, that's amazing! So this was actually revolutionary for its time?",
            "I'm so fascinated by this! What other techniques did they use?",
            "That's absolutely beautiful! I could study this for hours.",
            "Oh wow, this is so cool! I love learning about the history behind it.",
            "This is incredible! I'm learning so much today.",
            
            # Enthusiastic acknowledgments with personal connection
            "I love that! It reminds me of something I saw once but I can't quite place it.",
            "That's exactly what I was hoping to learn today! What else can you tell me?",
            "Oh this is wonderful! I'm so glad we stopped here. Keep going!",
            "That's so interesting! My grandmother used to tell me about art like this.",
            "Wow, I could look at this for hours! There's so much going on.",
            "I'm really enjoying this! This is exactly the kind of detail I love.",
            "Oh this is perfect! I've always wanted to understand this better.",
            "That's so cool! I'm going to remember this.",
            "I love this! It's making me see art in a whole new way.",
            "This is wonderful! I feel like I'm really understanding the artist's vision.",
            
            # Excited questions showing deep engagement
            "So does that mean this was actually painted for a specific person? Who were they?",
            "Wait, so the blue pigment was more expensive than gold? That's wild!",
            "That's amazing context! How did this painting end up in this museum?",
            "I love learning about this! Was this artist famous during their lifetime?",
            "That's so cool! Were there female artists doing this kind of work too?",
            "Oh interesting! What was the reaction when this was first shown?",
            "That's fascinating! How long did something like this take to create?",
            "I'm curious - did the artist have any students or followers?",
            "That's amazing! What inspired the artist to create this?",
            "Oh wow! Are there other pieces here by the same artist?",
            
            # Follow-up requests showing sustained interest
            "Tell me more! I want to know everything about this piece.",
            "Oh keep going! This is exactly the kind of stuff I love learning about.",
            "Don't stop there! What happened next?",
            "Yes! And? I'm totally hooked on this story now.",
            "Please continue! This is so interesting!",
            "I want to hear more! This is fascinating!",
            "Keep going! I'm really enjoying this!",
            "Tell me everything! I'm so curious!",
        ],
        
        "after_explain": [
            # Responsive to ExplainNewFact - show curiosity and follow-up interest
            "Oh that's so interesting! Can you tell me more about that?",
            "Wow, I never knew that! How does that relate to what we saw earlier?",
            "That's fascinating! What else should I know about this?",
            "I love learning that! Is there more to the story?",
            "That's incredible! Can you explain how that works?",
            "Oh wow, that's amazing! What does that mean for the painting?",
            "That's so cool! I want to understand this better.",
            "Fascinating! How did that influence the artist's work?",
            "I'm really intrigued by that! Can you elaborate?",
            "That's wonderful! What's the significance of that detail?",
            "Oh interesting! How does that connect to other pieces?",
            "That's remarkable! I'd love to learn more about that.",
            "Wow, that's so insightful! What else can you share?",
            "I'm fascinated! Can you tell me more about the technique?",
            "That's amazing! How did they achieve that effect?",
            "Oh I love that detail! What's the story behind it?",
            "That's incredible! I want to know everything about this.",
            "Fascinating! How does that relate to the time period?",
            "I'm so curious now! Can you explain more?",
            "That's wonderful! What makes that so special?",
        ],
        
        "after_question": [
            # Responsive to AskOpinion/AskMemory - thoughtful, engaged responses
            "Oh that's a great question! Let me think... I'd say what really strikes me is the emotion in the faces.",
            "Hmm, interesting question! I think what I notice most is how realistic everything looks.",
            "That's a thoughtful question! I'm drawn to the way the light plays across the scene.",
            "Oh I love that you asked! I think the colors are what really stand out to me.",
            "That's such a good question! I find myself looking at the composition - it's so balanced.",
            "Hmm, let me think about that... I think what catches my eye is the level of detail.",
            "Oh that's interesting to consider! I'd say the mood is what really gets me.",
            "I appreciate you asking! I think the technique is what amazes me most.",
            "That's a deep question! I'm really struck by the artist's skill here.",
            "Oh I'm glad you asked! I think what I love most is the storytelling aspect.",
            "That's such an insightful question! I notice the way everything seems to come together.",
            "Hmm, that makes me think... I'd say the historical context is fascinating.",
            "Oh that's a wonderful question! I think the craftsmanship is what impresses me.",
            "I love that question! I'm really drawn to the way the artist captured the moment.",
            "That's a thoughtful way to look at it! I think the symbolism is what interests me.",
        ],
    },
    
    # =========================================================================
    # ENGAGED - Natural acknowledgment with occasional curiosity
    # Target: 100+ templates total
    # =========================================================================
    
    "ENGAGED": {
        "general": [
            # Natural acknowledgments
            "Oh interesting, I didn't know that. Makes sense though.",
            "Huh, that's cool. I can see what you mean now.",
            "Ah I see. Yeah, that really shows in the painting.",
            "Nice, that's good to know. What else should I notice?",
            "Oh okay, that's interesting. I never thought about it that way.",
            "Right, I can see that now that you point it out.",
            "That makes sense. The artist really knew what they were doing.",
            "Cool, I like learning about the history behind it.",
            "Interesting. Is that typical for this time period?",
            "Oh neat, I appreciate you explaining that.",
            "That's helpful. I'm starting to see the bigger picture.",
            "Ah okay, that clarifies things. Thanks for that.",
            "Right, that makes a lot of sense actually.",
            "I see what you mean. That's a good point.",
            "Oh interesting perspective. I hadn't considered that.",
            "That's good to know. I'll look for that next time.",
            "Ah, that explains a lot. Makes sense now.",
            "Right, I can definitely see that in the work.",
            "That's helpful context. I appreciate the explanation.",
            "Oh I see. That's really interesting actually.",
            
            # Mild curiosity without overwhelming enthusiasm
            "Hm, that's actually pretty cool. How did they do that exactly?",
            "I see, so it was intentional then. Interesting choice.",
            "Oh, that's a nice detail. I wouldn't have noticed it myself.",
            "Got it, that helps. The colors are really nice up close.",
            "Ah okay, thanks for explaining. I was wondering about that.",
            "Yeah, I can tell a lot of effort went into this.",
            "That's helpful context. I like knowing the backstory.",
            "Oh I see. So there's meaning behind those details?",
            "Makes sense. It's more complex than it first appears.",
            "Interesting. You can really tell it's from that era.",
            "Hmm, that's interesting. I hadn't thought about it that way.",
            "Oh okay, that makes sense. I can see the connection now.",
            "Right, I understand. That's a good way to look at it.",
            "That's helpful. I'm learning a lot here.",
            "Ah, that's interesting. I appreciate you sharing that.",
            "I see. That adds another layer to understanding it.",
            "Oh that's cool. I like how everything connects.",
            "That makes sense. I can see why it's significant.",
            "Interesting point. I hadn't noticed that before.",
            "Oh I get it. That's really helpful context.",
            
            # Brief engaged responses
            "Oh nice, I like that.",
            "Cool, good to know.",
            "Ah, that's neat.",
            "Right, I can see it.",
            "Interesting, thanks.",
            "That's nice.",
            "I see.",
            "Got it.",
            "Okay, thanks.",
            "That's helpful.",
            "Nice to know.",
            "I appreciate that.",
            "That's interesting.",
            "Good to know.",
            "Thanks for explaining.",
        ],
        
        "after_explain": [
            # Responsive to ExplainNewFact - show understanding and mild curiosity
            "Oh that's interesting. I can see how that would matter.",
            "Right, that makes sense. I hadn't thought about it that way.",
            "Ah okay, that's helpful. I'm starting to understand better.",
            "That's good to know. I'll keep that in mind.",
            "Oh I see. That adds context to what we're looking at.",
            "Interesting. I can see why that detail is important.",
            "That's helpful. I'm learning a lot about this piece.",
            "Oh okay, that makes sense. Thanks for explaining.",
            "Right, I can see that now. That's interesting.",
            "That's good context. I appreciate you sharing that.",
            "Oh I understand. That's a nice detail to know.",
            "Interesting point. I hadn't considered that aspect.",
            "That's helpful information. I'm getting a better picture.",
            "Oh I see what you mean. That makes sense.",
            "Right, that's interesting. I can see the connection.",
        ],
        
        "after_question": [
            # Responsive to AskOpinion/AskMemory - thoughtful but not overly enthusiastic
            "Hmm, let me think... I'd say what catches my eye is the composition.",
            "Oh that's a good question. I think I'm drawn to the colors mostly.",
            "That's interesting to consider. I notice the level of detail.",
            "Hmm, I'd say what stands out is how realistic it looks.",
            "Oh I like that question. I think the mood is what gets me.",
            "That makes me think... I'd say the technique is impressive.",
            "Oh that's thoughtful. I think what I notice is the storytelling.",
            "Hmm, let me see... I'd say the craftsmanship is what stands out.",
            "That's a nice way to look at it. I think the emotion is powerful.",
            "Oh I appreciate you asking. I'd say the historical context is interesting.",
            "That's a good question. I think what draws me in is the detail.",
            "Hmm, I'd say I'm really struck by how everything comes together.",
            "Oh that's interesting. I think the artist's skill is what amazes me.",
            "That makes me consider... I'd say the symbolism is fascinating.",
            "Oh I like thinking about that. I think the overall effect is striking.",
        ],
    },
    
    # =========================================================================
    # CONFUSED - Specific confusion that needs articulation
    # Target: 60+ templates total
    # =========================================================================
    
    "CONFUSED": {
        "general": [
            # Confusion about timeline/history
            "Wait, I'm a bit lost - you said this was painted in the 1600s but also mentioned something about a commission?",
            "Sorry, I didn't follow that. When exactly was this made? I got confused by the dates.",
            "Hmm, I'm not sure I understand. Was the artist Dutch or was the subject Dutch?",
            "Hold on, I missed something. This was before or after the other painting you mentioned?",
            "Sorry, you lost me there. Can you explain what you meant about the time period?",
            "Wait, I'm confused about the timeline. Can you clarify when this happened?",
            "I'm not following - you mentioned dates but I got lost. Can you help?",
            "Sorry, I'm confused about the sequence of events. Can you go over that again?",
            "Hold on, I think I missed something about the history. Can you clarify?",
            "I'm a bit lost on the timeline. When exactly did this happen?",
            
            # Confusion about technique/art terms
            "I don't think I get that. What does that technique actually mean in plain terms?",
            "Wait, what? Can you explain that differently? I'm not familiar with art terminology.",
            "Hmm, that's confusing. What's the difference between what you just said and the regular way?",
            "Sorry, I'm confused - what do you mean by that? I don't know much about painting.",
            "I didn't quite catch that. Could you put it in simpler terms?",
            "Wait, I'm not sure what that means. Can you explain it like I'm new to this?",
            "I'm confused about the technique. What does that actually involve?",
            "Sorry, I don't understand the art terms. Can you use simpler language?",
            "I'm lost on what that means. Could you break it down for me?",
            "Wait, what's the difference? I'm not following the technical details.",
            
            # Confusion about significance/meaning
            "Okay I'm a bit confused - why is that detail important exactly?",
            "Wait, I don't get the connection. How does that relate to what you said before?",
            "I'm lost - why would the artist do that? Seems like an odd choice.",
            "Sorry, I don't understand. What makes that different from other paintings?",
            "Can you clarify? I'm not seeing what you're describing.",
            "I'm confused about why that matters. Can you explain the significance?",
            "Wait, I don't understand the connection. How does that work?",
            "I'm lost on why that's important. Can you help me understand?",
            "Sorry, I'm not seeing the relevance. Can you clarify?",
            "I'm confused about the meaning. What are you trying to say?",
            
            # General confusion with request for help
            "Hmm, I think I missed something. Can you go over that again?",
            "Sorry, my mind wandered. What were you saying about the colors?",
            "I'm confused - can we back up a bit? I want to make sure I understand.",
            "Wait, I'm lost. Can you start over with that explanation?",
            "I think I missed something. Can you repeat that last part?",
            "Sorry, I got confused. Can you explain that one more time?",
            "I'm not following. Can you break it down differently?",
            "Wait, I'm confused. Can you clarify what you meant?",
            "I think I need that explained again. I didn't quite get it.",
            "Sorry, I'm lost. Can you help me understand?",
        ],
        
        "after_clarify": [
            # Responsive to ClarifyFact - show relief or continued confusion
            "Oh okay, I think I'm starting to get it now. Thanks for explaining differently.",
            "Hmm, that helps a bit. I'm still a little confused though.",
            "Oh I see what you mean now. That makes more sense.",
            "Okay, I think I understand better. Thanks for being patient.",
            "Oh that's clearer. I appreciate you explaining it another way.",
            "Hmm, I'm still not entirely sure. Can you give me another example?",
            "Oh okay, that helps. I think I'm getting there.",
            "I see, that's a better way to put it. I understand now.",
            "Oh that makes more sense. Thanks for clarifying.",
            "Hmm, I think I need one more explanation. I'm still confused.",
        ],
    },
    
    # =========================================================================
    # CURIOUS - Questions with context, genuine inquiry
    # Target: 70+ templates total
    # =========================================================================
    
    "CURIOUS": {
        "general": [
            # Questions about the artist
            "So what do we know about the artist? Were they well-known during their time?",
            "I'm curious - did the artist make other paintings like this one?",
            "What was the artist trying to convey here? Is there a deeper meaning?",
            "Do we know why the artist chose this particular subject?",
            "Was this artist part of a movement or did they work independently?",
            "What was the artist's background? Where did they train?",
            "I'm curious about the artist's life. What was their story?",
            "Did the artist have any particular influences? Who inspired them?",
            "What was the artist known for? What made them famous?",
            "I wonder about the artist's technique. How did they develop their style?",
            
            # Questions about technique
            "How did they achieve that effect with the light? It looks so realistic.",
            "What kind of materials did artists use back then? Same as today?",
            "I wonder how long something like this took to paint. Any idea?",
            "Is this style of painting difficult to do? It looks incredibly detailed.",
            "What makes this particular technique special compared to others?",
            "How did they create that texture? It looks so intricate.",
            "I'm curious about the brushwork. What technique did they use?",
            "How did they mix the colors? Did they have special methods?",
            "What tools did artists use back then? Were they different from now?",
            "I wonder about the process. How did they plan this out?",
            
            # Questions about history/context
            "What was happening in the world when this was made? Was it a peaceful time?",
            "Who would have owned a painting like this originally?",
            "How did people back then react to art like this? Was it controversial?",
            "What's the story behind this scene? Is it depicting something specific?",
            "Were there specific rules artists had to follow back then?",
            "What was the art world like during this period?",
            "I'm curious about the historical context. What was life like then?",
            "How did paintings like this fit into society at the time?",
            "What was the purpose of this painting? Was it for display or private?",
            "I wonder about the cultural significance. What did this represent?",
            
            # Questions about the museum/collection
            "How long has the museum had this piece?",
            "Is this one of the more famous pieces in the collection?",
            "What makes this museum special for Dutch art?",
            "Are there other paintings by the same artist here?",
            "How do they keep these old paintings in such good condition?",
            "What's the history of this piece in the museum?",
            "I'm curious - how did the museum acquire this?",
            "Are there other similar pieces in the collection?",
            "What's the significance of having this piece here?",
            "I wonder about the restoration. Has this been worked on?",
            
            # General curiosity questions
            "I'm curious - what should I be looking for in this piece?",
            "What makes this painting stand out? What's special about it?",
            "I wonder about the symbolism. Is there hidden meaning?",
            "What's the most interesting thing about this piece?",
            "I'm curious - what do experts find most fascinating here?",
        ],
        
        "after_explain": [
            # Responsive to ExplainNewFact - deeper follow-up questions
            "Oh that's interesting! How does that relate to what you mentioned before?",
            "I'm curious - what else should I know about that?",
            "That's fascinating! Can you tell me more about how that works?",
            "Oh interesting! What's the significance of that detail?",
            "I'm really curious now - how did that influence the artist?",
            "That's intriguing! What else connects to that?",
            "Oh I want to know more! Can you elaborate on that?",
            "That's so interesting! How does that fit into the bigger picture?",
            "I'm curious - what's the story behind that?",
            "Oh that's fascinating! What else can you tell me?",
        ],
    },
    
    # =========================================================================
    # OVERLOADED - Trying to process, feeling overwhelmed
    # Target: 40+ templates total
    # =========================================================================
    
    "OVERLOADED": {
        "general": [
            "Okay, okay... that's a lot to take in. Give me a second.",
            "Right, right. I think I'm following. Maybe slow down a bit?",
            "Yeah... okay. There's so much to remember here.",
            "Alright, I think I got most of that. Probably.",
            "Okay, that's a lot of information. Let me process that.",
            "Sure, sure. I'm trying to keep up here.",
            "Got it... I think. This is a lot more complex than I expected.",
            "Okay okay, hold on. Let me catch up mentally.",
            "Right, that makes sense... I think? There's just so much.",
            "Yeah okay. My brain is working overtime right now.",
            "Mm, okay. I might need a moment with this one.",
            "Alright, I'm trying to process all of this.",
            "Okay, that's a lot. Can we slow down a little?",
            "Right, I think I'm following. It's just a lot.",
            "Yeah, okay. I'm taking it all in.",
            "Okay okay, I'm processing. Give me a sec.",
            "Right, I think I got it. There's just so much information.",
            "Yeah, I'm trying to keep up. It's a lot.",
            "Okay, I'm following. Just need to process.",
            "Right right, I think I understand. It's complex though.",
            "Yeah okay. My head is spinning a bit.",
            "Alright, I'm trying to absorb all this.",
            "Okay, that's a lot to digest. Let me think.",
            "Right, I'm following. Just a lot to take in.",
            "Yeah, I think I'm getting it. It's just overwhelming.",
        ],
    },
    
    # =========================================================================
    # FATIGUED - Tired, minimal energy, short responses
    # Target: 30+ templates total
    # =========================================================================
    
    "FATIGUED": {
        "general": [
            "Mm-hmm.",
            "Yeah.",
            "Okay.",
            "Right.",
            "Sure.",
            "Uh-huh.",
            "Mhm.",
            "Yep.",
            "Got it.",
            "Okay, yeah.",
            "Mm.",
            "Yeah, nice.",
            "*nods*",
            "Cool.",
            "Right, okay.",
            "Sure, sure.",
            "Mm-hmm, okay.",
            "Yeah, I see.",
            "Okay.",
            "Right.",
            "Mm.",
            "Yeah.",
            "Okay, got it.",
            "Sure.",
            "Mm-hmm.",
            "Right.",
            "Yeah, okay.",
            "Okay.",
            "Mm.",
            "Sure.",
        ],
    },
    
    # =========================================================================
    # READY_TO_MOVE - Wants to see something else
    # Target: 35+ templates total
    # =========================================================================
    
    "READY_TO_MOVE": {
        "general": [
            "This has been great! I think I'm ready to see something else now though.",
            "I feel like I've got a good sense of this one. What's next?",
            "Okay I think I've seen enough here. What else do you recommend?",
            "This was really interesting! I'm curious what else is in this gallery.",
            "Nice! Shall we move on? I want to make sure we see everything.",
            "I think I'm good here. Is there something nearby we should check out?",
            "Alright, I feel like I've absorbed this one. Ready for the next!",
            "Great explanation! I'm excited to see what else you want to show me.",
            "I've got a good picture now. What would you suggest we see next?",
            "I think I've learned enough about this one. What's next?",
            "This has been wonderful! Ready to explore something new.",
            "I feel like I understand this piece now. Shall we continue?",
            "Great! I'm ready to see what else you have to show.",
            "I think I've got a good sense of this. What's next?",
            "This was interesting! I'm ready to move on.",
            "I've enjoyed this! What else should we see?",
            "I think I'm ready for something new. What do you recommend?",
            "This has been lovely! Ready to see more.",
            "I feel like I've seen enough here. What's next?",
            "Great! I'm excited to see what else is here.",
            "I think I've got it. Ready for the next piece!",
            "This was great! What else can we explore?",
            "I'm ready to move on. What should we see next?",
            "I've learned a lot! Ready for something new.",
            "This has been perfect! What's next?",
            "I think I'm good here. Ready to continue!",
            "Great explanation! I'm ready for more.",
            "I've got it now. What else should we see?",
            "This was wonderful! Ready to explore further.",
            "I think I understand this one. What's next?",
            "Perfect! I'm ready to see what else you have.",
            "I've enjoyed this! Ready to move forward.",
            "This has been great! What's our next stop?",
            "I'm ready! What else should we explore?",
            "Great! I'm excited to see more.",
        ],
    },
    
    # =========================================================================
    # BORED_OF_TOPIC - Subtle redirection, impatience
    # Target: 30+ templates total
    # =========================================================================
    
    "BORED_OF_TOPIC": {
        "general": [
            "Yeah, I think I've got the gist. What's over in that corner?",
            "Okay okay, I get it. Is there anything here that's really different?",
            "Right right. Hey, are there any sculptures or just paintings?",
            "Got it. I feel like I've seen a lot of portraits though - anything else?",
            "Mm-hmm, okay. What about something more modern? Or colorful maybe?",
            "Sure sure. Is there anything interactive here or all just looking?",
            "Yeah got it. What's the most popular piece in this museum anyway?",
            "Okay, that's enough about this one I think. Show me something surprising.",
            "Right, I understand. What else is there to see?",
            "Yeah okay. I'm kind of curious about something different.",
            "Got it. Anything else that's really different?",
            "Okay, I think I've got this. What's next?",
            "Right right. What else is interesting here?",
            "Yeah, I get it. Show me something else.",
            "Okay okay. What's over there?",
            "Got it. I want to see something different.",
            "Right. What else can we look at?",
            "Yeah, I understand. What's next?",
            "Okay, I've got it. What else?",
            "Right. Show me something new.",
            "Yeah okay. What's different?",
            "Got it. What else?",
            "Okay. Next?",
            "Right. What else?",
            "Yeah. Something else?",
            "Okay. What's next?",
            "Right. Different?",
            "Yeah. More?",
            "Okay. Next piece?",
            "Right. What else?",
        ],
    },
    
    # =========================================================================
    # DISENGAGED - Checked out, minimal engagement
    # Target: 25+ templates total
    # =========================================================================
    
    "DISENGAGED": {
        "general": [
            "Mm-hmm.",
            "Yeah.",
            "Sure.",
            "Okay.",
            "Uh-huh.",
            "Right.",
            "*looking at phone*",
            "...",
            "Mm.",
            "K.",
            "Yep.",
            "*glances around*",
            "Mm-hmm.",
            "Yeah.",
            "Sure.",
            "Okay.",
            "Right.",
            "Mm.",
            "Uh-huh.",
            "Yep.",
            "...",
            "Mm-hmm.",
            "Yeah.",
            "Okay.",
            "Sure.",
        ],
    },
}


# =============================================================================
# SIM8 RESPONSE TYPE TEMPLATES - Expanded with responsiveness
# =============================================================================

SIM8_RESPONSES = {
    "question": {
        "general": [
            # Questions about specific artwork elements
            "So what's the story behind this particular detail here?",
            "Can you tell me more about why the artist chose those colors?",
            "What does this symbolize exactly? I feel like there's meaning I'm missing.",
            "Who was this painted for originally? Was it a commission?",
            "When was this made and what was happening at the time?",
            "How did they achieve that effect with the light? It looks so natural.",
            "Is there a reason this is considered so important?",
            "What technique was used here? It looks different from others I've seen.",
            "Were there other artists doing similar work at this time?",
            "What happened to this painting before it came to the museum?",
            "What's the significance of this particular scene?",
            "How did the artist create that texture? It's so detailed.",
            "What was the inspiration behind this piece?",
            "I'm curious - what makes this painting special?",
            "Can you explain the symbolism here?",
            "What was the artist trying to convey?",
            "How long did this take to paint?",
            "What materials did they use?",
            "Who would have owned this originally?",
            "What's the historical context for this?",
            
            # Broader curiosity questions
            "What should I be paying attention to here?",
            "Is there something special about this one I should know about?",
            "Can you explain what makes this style distinctive?",
            "I'm curious - how do you know all these details?",
            "What's the most interesting thing about this piece?",
            "What should I notice that I might miss?",
            "What makes this different from other paintings?",
            "What's the story here?",
            "What's fascinating about this?",
            "What should I know about this?",
            "What's unique about this piece?",
            "What's the background here?",
            "What's interesting about this?",
            "What should I look for?",
            "What's special here?",
            "What's the context?",
            "What's the history?",
            "What's notable?",
            "What's important?",
            "What's the deal?",
        ],
        
        "after_explain": [
            # Responsive to ExplainNewFact - follow-up questions
            "Oh that's interesting! How does that work?",
            "That's fascinating! What else should I know?",
            "I'm curious - can you tell me more about that?",
            "Oh interesting! How does that relate?",
            "That's cool! What's the significance?",
            "I want to know more! Can you elaborate?",
            "That's intriguing! How did that happen?",
            "Oh I'm curious! What else connects to that?",
            "That's fascinating! Can you explain more?",
            "I'm really interested! Tell me more.",
        ],
    },
    
    "statement": {
        "general": [
            # Appreciative statements
            "The colors in this are really beautiful. I love how warm it feels.",
            "You can really see the skill that went into this. Impressive.",
            "I like how detailed everything is. Must have taken forever.",
            "This reminds me of something I saw once but I can't quite place it.",
            "The way they captured the light is really something special.",
            "I appreciate how realistic this looks. The faces seem alive.",
            "There's something calming about this painting. I like the mood.",
            "The composition is really well balanced. My eye keeps moving around.",
            "I can see why this would be famous. It has a certain presence.",
            "It's interesting how different it looks up close versus far away.",
            "The craftsmanship here is really impressive. You can see the care.",
            "I love the way the colors work together. It's harmonious.",
            "This has such a strong emotional impact. It's moving.",
            "The detail is incredible. Every part is carefully considered.",
            "I appreciate the artistry here. It's really well done.",
            "This is beautiful. I could look at it for a while.",
            "The technique is remarkable. So much skill on display.",
            "I like how the artist captured the moment. It feels alive.",
            "This is really well executed. The quality is evident.",
            "I'm impressed by the level of detail. It's meticulous.",
            
            # Personal connection statements
            "This actually makes me want to learn more about this period.",
            "I've always liked Dutch art but never really knew why until now.",
            "Art like this makes me appreciate what humans can create.",
            "This connects with me. I feel like I understand something new.",
            "I'm really enjoying this. It's opening my eyes.",
            "This makes me think. I like art that does that.",
            "I appreciate this more now. Thanks for the context.",
            "This is meaningful to me. I'll remember this.",
            "I feel like I'm seeing something special here.",
            "This resonates with me. I understand why it's important.",
            "I'm really getting into this. It's fascinating.",
            "This is making me think differently. I like that.",
            "I appreciate you sharing this. It's enriching.",
            "This is beautiful. I'm glad I'm seeing it.",
            "I'm really enjoying learning about this.",
            "This is special. I can feel it.",
            "I'm connecting with this piece.",
            "This means something to me.",
            "I'm really appreciating this.",
            "This is wonderful.",
        ],
    },
    
    "confusion": {
        "general": [
            "Wait, I'm not sure I follow. Can you explain that differently?",
            "Sorry, I got lost there. What did you mean exactly?",
            "Hmm, I don't think I understand. What's the significance of that?",
            "I'm confused about what you said. Can you break it down?",
            "Could you clarify? I'm not familiar with that term.",
            "I missed something. How does that relate to what we're looking at?",
            "Sorry, I don't get the connection. Why does that matter?",
            "Can you go over that again? I want to make sure I understand.",
            "I'm not following. Can you explain it simpler?",
            "Wait, I'm confused. What does that mean?",
            "I don't understand. Can you help me?",
            "I'm lost. Can you clarify?",
            "I'm not getting it. Can you explain?",
            "I'm confused. What are you saying?",
            "I don't follow. Can you help?",
            "I'm not sure. Can you explain?",
            "I'm lost. Help me understand.",
            "I don't get it. Explain?",
            "I'm confused. Clarify?",
            "I don't understand.",
            "I'm lost.",
            "I'm confused.",
            "I don't get it.",
            "I'm not following.",
            "I don't understand.",
            "I'm lost.",
            "I'm confused.",
            "I don't get it.",
            "I'm not sure.",
            "I'm lost.",
            "I'm confused.",
            "I don't understand.",
            "I'm lost.",
            "I'm confused.",
            "I don't get it.",
            "I'm not following.",
            "I don't understand.",
            "I'm lost.",
            "I'm confused.",
            "I don't get it.",
        ],
    },
    
    "acknowledgment": {
        "general": [
            "Oh that's really interesting! I didn't know that before.",
            "Ah I see, that makes a lot of sense now. Thanks for explaining.",
            "Oh okay, that helps me appreciate it more. Good to know.",
            "Right, I can definitely see that now that you point it out.",
            "That's fascinating context. It changes how I see it.",
            "Ah got it. I love learning the stories behind things.",
            "Oh interesting! That's exactly the kind of detail I was hoping to hear.",
            "Thanks for explaining that. I wouldn't have figured it out myself.",
            "Oh I see, that's really helpful. I'll look for that in other paintings.",
            "That's helpful. I'm understanding this better now.",
            "Oh I get it. That makes sense.",
            "Right, that's interesting. Thanks.",
            "Oh okay, I see. That helps.",
            "That's good to know. I appreciate it.",
            "Oh I understand. That's helpful.",
            "Right, that makes sense. Thanks.",
            "Oh I see. That's interesting.",
            "That's helpful. I get it now.",
            "Oh okay. Thanks for that.",
            "Right, I understand. Thanks.",
            "Oh I see. Good to know.",
            "That's helpful. Thanks.",
            "Oh I get it. Thanks.",
            "Right, thanks.",
            "Oh I see. Thanks.",
            "That's helpful.",
            "Oh I get it.",
            "Right, thanks.",
            "Oh I see.",
            "Thanks.",
            "Oh I get it.",
            "Right.",
            "Oh I see.",
            "Thanks.",
            "Oh I get it.",
            "Right.",
            "Oh I see.",
            "Thanks.",
            "Oh I get it.",
            "Right.",
            "Oh I see.",
        ],
    },
    
    "follow_up_question": {
        "general": [
            "And what happened after that? I want to know the rest of the story.",
            "So does that mean there were other artists influenced by this?",
            "Is that why the other paintings from this period look similar?",
            "Wait, so how does that connect to what you told me earlier?",
            "Interesting - are there more examples of this technique here?",
            "Does that apply to all Dutch paintings or just this style?",
            "So did that change how artists worked after that?",
            "And is that typical for this artist or was it unusual?",
            "What happened next? I want to know more.",
            "How does that relate to what we discussed?",
            "Can you tell me more about that?",
            "What else connects to that?",
            "How does that work?",
            "What's the connection?",
            "How does that relate?",
            "What else?",
            "Tell me more.",
            "What happened?",
            "How does that work?",
            "What's next?",
            "And then?",
            "What else?",
            "Tell me more.",
            "What happened?",
            "How?",
            "What?",
            "And?",
            "More?",
            "Next?",
            "Then?",
            "What?",
            "How?",
            "And?",
            "More?",
            "Next?",
            "Then?",
        ],
    },
    
    "reference": {
        "general": [
            "Going back to what you said about the composition - can you explain more?",
            "You mentioned the symbolism earlier. What did you mean by that?",
            "About the historical context - I'm curious to hear more.",
            "Can we talk more about the technique you mentioned?",
            "I'm still thinking about what you said about the artist's life.",
            "That thing you mentioned about the commission was interesting.",
            "You said something about the colors earlier. Can you elaborate?",
            "I'm thinking about what you mentioned before. Can we go back to that?",
            "That detail you mentioned - I want to understand it better.",
            "Going back to what you said - can you explain more?",
            "You mentioned something earlier. Can you tell me more?",
            "I'm curious about what you said before.",
            "Can we return to that topic?",
            "You mentioned that earlier.",
            "Going back to that.",
            "About what you said.",
            "You mentioned that.",
            "Going back.",
            "About that.",
            "You said.",
            "Earlier.",
            "Before.",
            "That.",
            "It.",
            "What you said.",
        ],
    },
    
    "repeat_request": {
        "general": [
            "Sorry, what was that last part? I want to make sure I caught it.",
            "Could you repeat that? I didn't quite hear you.",
            "What was the artist's name again? I want to remember it.",
            "Sorry, I missed the date you mentioned. When was it?",
            "Can you say that again? I want to make sure I understood.",
            "What was that? I didn't catch it.",
            "Can you repeat that? I missed it.",
            "What did you say? I didn't hear.",
            "Sorry, what was that?",
            "Can you say that again?",
            "What was that?",
            "Repeat that?",
            "Say again?",
            "What?",
            "Again?",
            "Sorry?",
            "What?",
            "Again?",
            "Repeat?",
            "What?",
        ],
    },
    
    "silence": {
        "general": [
            "",
        ],
    },
}


# =============================================================================
# TRANSITION TEMPLATES - For OfferTransition responses
# =============================================================================

# Templates for when visitor ACCEPTS the transition offer
# {exhibit} will be replaced with the target exhibit name
TRANSITION_ACCEPTANCE_TEMPLATES = [
    "Yes, let's go see {exhibit}! I'd love to check that out.",
    "Oh yes! {exhibit} sounds interesting. Lead the way!",
    "Sure, let's head over to {exhibit}. I'm curious about it.",
    "Great idea! I'd love to see {exhibit} next.",
    "Yes please! {exhibit} sounds fascinating.",
    "Oh, {exhibit}? Yes, let's go see that!",
    "That sounds great! I've been wanting to see {exhibit}.",
    "Let's do it! {exhibit} looks really interesting.",
    "Perfect! I'm ready to move on to {exhibit}.",
    "Yes! I'd love to see what {exhibit} is all about.",
    "Okay, let's check out {exhibit}. I'm excited!",
    "Sure thing! {exhibit} is exactly what I want to see next.",
    "Oh wonderful! Let's go to {exhibit}.",
    "{exhibit}? Yes, I've been curious about that one!",
    "Sounds good! I'm ready to see {exhibit}.",
    "Absolutely! {exhibit} sounds like a great next stop.",
    "Yes, let's! I can't wait to see {exhibit}.",
    "Oh perfect! {exhibit} was on my list to see.",
    "Great suggestion! {exhibit} it is!",
    "I'd love that! Let's head to {exhibit}.",
    "Yes! {exhibit} sounds wonderful.",
    "Oh, that's a great idea! Let's go see {exhibit}.",
    "Sure! {exhibit} looks amazing.",
    "Let's go! {exhibit} sounds fascinating.",
    "Wonderful! {exhibit} next.",
]

# Templates for when visitor REJECTS the transition offer
# {exhibit} is optional - may or may not be used
TRANSITION_REJECTION_TEMPLATES = [
    "Wait, I want to learn more about this first.",
    "Hold on, can you tell me more before we move?",
    "Actually, I'm still curious about this.",
    "I'd like to stay here a bit longer.",
    "Not yet, I have more questions about this one.",
    "Can we stay here a little longer? I'm still interested.",
    "Hmm, I'm not quite ready to move on yet.",
    "Actually, I'd rather stay here for now.",
    "I feel like there's more to see here.",
    "Let's not rush off just yet.",
    "Maybe in a bit? I'm still taking this in.",
    "I'm not finished looking at this one.",
    "Can we explore this a bit more first?",
    "Hold on, I still have questions about this.",
    "I think I'd like to stay here a bit longer.",
    "Not just yet, I want to understand this better.",
    "Let's stay here for now. This is interesting.",
    "I'm not ready to leave this one yet.",
    "Actually, I'd like to hear more about this first.",
    "Can you tell me more about this before we go?",
]


# =============================================================================
# RESPONSIVENESS LOGIC
# =============================================================================

def _select_visitor_template_category(
    state_or_type: str,
    agent_option: Optional[str],
    agent_subaction: Optional[str],
    rng: random.Random
) -> str:
    """
    Select template category based on agent action for responsiveness.
    
    Returns:
        Category name: "general", "after_explain", "after_question", "after_clarify", etc.
    """
    # Check if we have categorized templates for this state/type
    if isinstance(VISITOR_RESPONSES.get(state_or_type), dict):
        # We have categorized templates
        if agent_subaction == "ExplainNewFact":
            if rng.random() < 0.4:  # 40% chance for responsive template
                if "after_explain" in VISITOR_RESPONSES[state_or_type]:
                    return "after_explain"
        elif agent_subaction in ("AskOpinion", "AskMemory", "AskClarification"):
            if rng.random() < 0.3:  # 30% chance for responsive template
                if "after_question" in VISITOR_RESPONSES[state_or_type]:
                    return "after_question"
        elif agent_subaction == "ClarifyFact":
            if rng.random() < 0.35:  # 35% chance for responsive template
                if "after_clarify" in VISITOR_RESPONSES[state_or_type]:
                    return "after_clarify"
    
    # Default to general
    return "general"


# =============================================================================
# VARIETY ENHANCEMENT FUNCTIONS
# =============================================================================

def add_natural_variation(text: str, rng: random.Random) -> str:
    """Add slight variations to make responses feel more natural."""
    
    # 10% chance to add a thinking pause at start
    if rng.random() < 0.10 and not text.startswith(("*", "Hmm", "Um", "Well")):
        pauses = ["Hmm, ", "Well, ", "Let me think... ", ""]
        text = rng.choice(pauses) + text[0].lower() + text[1:] if text else text
    
    # 8% chance to add trailing thought
    if rng.random() < 0.08 and text and not text.endswith(("?", "...")) and len(text) > 10:
        trails = ["..", " I think.", " I guess.", ""]
        text = text.rstrip(".!") + rng.choice(trails)
    
    return text


def clean_utterance(text: str) -> str:
    """Clean up the generated utterance."""
    # Remove double spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Fix double punctuation
    text = re.sub(r'([.!?])\1+', r'\1', text)
    text = re.sub(r'\.\?', '?', text)
    text = re.sub(r'\?\.', '?', text)
    text = re.sub(r'\.\.\.+', '...', text)
    
    return text


# =============================================================================
# MAIN GENERATION FUNCTION
# =============================================================================

def generate_visitor_utterance(
    response_type: Optional[str],
    visitor_state: Optional[str],
    aoi: str,
    rng: random.Random,
    agent_option: Optional[str] = None,
    agent_subaction: Optional[str] = None,
    transition_success: bool = False,
    transition_rejected: bool = False,
    target_exhibit: Optional[str] = None
) -> str:
    """
    Generate varied visitor utterance without LLM.
    
    Args:
        response_type: Sim8 response type (question, statement, confusion, etc.)
        visitor_state: StateMachine state (ENGAGED, CONFUSED, CURIOUS, etc.)
        aoi: Current exhibit/AOI name for context
        rng: Seeded random generator for reproducibility
        agent_option: Agent's option (Explain, AskQuestion, etc.) for responsiveness
        agent_subaction: Agent's subaction (ExplainNewFact, AskOpinion, etc.) for responsiveness
        transition_success: Whether a transition offer was accepted
        transition_rejected: Whether a transition offer was rejected
        target_exhibit: Target exhibit name for transition responses
    
    Returns:
        Generated visitor utterance string
    """
    # Handle silence
    if response_type == "silence":
        return ""
    
    # === TRANSITION HANDLING ===
    # Check for transition acceptance/rejection FIRST (before general templates)
    if agent_option == "OfferTransition" or agent_subaction == "SuggestMove":
        # Format exhibit name nicely (replace underscores, title case)
        exhibit_name = target_exhibit or aoi or "that"
        exhibit_name = exhibit_name.replace("_", " ").title()
        
        if transition_success:
            # Visitor accepts the transition
            template = rng.choice(TRANSITION_ACCEPTANCE_TEMPLATES)
            utterance = template.format(exhibit=exhibit_name)
            return clean_utterance(utterance)
        elif transition_rejected:
            # Visitor rejects the transition
            template = rng.choice(TRANSITION_REJECTION_TEMPLATES)
            # Some templates don't use {exhibit}, so use safe formatting
            if "{exhibit}" in template:
                utterance = template.format(exhibit=exhibit_name)
            else:
                utterance = template
            return clean_utterance(utterance)
    
    # Determine which response pool to use
    state_or_type = None
    responses = None
    
    if visitor_state and visitor_state in VISITOR_RESPONSES:
        # StateMachine simulator - use visitor state
        state_or_type = visitor_state
        responses = VISITOR_RESPONSES[visitor_state]
    elif response_type and response_type in SIM8_RESPONSES:
        # Sim8 simulator - use response type
        state_or_type = response_type
        responses = SIM8_RESPONSES[response_type]
    else:
        # Fallback to engaged
        state_or_type = "ENGAGED"
        responses = VISITOR_RESPONSES["ENGAGED"]
    
    # Handle categorized templates (for responsiveness)
    if isinstance(responses, dict):
        # Select category based on agent action
        category = _select_visitor_template_category(
            state_or_type, agent_option, agent_subaction, rng
        )
        
        # Get templates from selected category, fallback to general
        if category in responses:
            template_list = responses[category]
        else:
            template_list = responses.get("general", [])
    else:
        # Old format - flat list (backward compatibility)
        template_list = responses
    
    # Select a random response
    if not template_list:
        # Ultimate fallback
        return "Interesting."
    
    utterance = rng.choice(template_list)
    
    # Apply natural variation
    utterance = add_natural_variation(utterance, rng)
    
    # Clean up
    utterance = clean_utterance(utterance)
    
    return utterance


# =============================================================================
# TESTING / DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    # Quick demonstration of variety
    rng = random.Random(42)
    
    print("=" * 70)
    print("VISITOR TEMPLATE V3.0 DEMONSTRATION")
    print("=" * 70)
    
    # Test StateMachine states
    print("\n--- StateMachine States ---")
    for state in ["HIGHLY_ENGAGED", "ENGAGED", "CONFUSED", "CURIOUS", "FATIGUED", "BORED_OF_TOPIC"]:
        if state in VISITOR_RESPONSES:
            responses = VISITOR_RESPONSES[state]
            if isinstance(responses, dict):
                total = sum(len(v) for v in responses.values())
            else:
                total = len(responses)
            print(f"\n{state} ({total} templates):")
            for _ in range(3):
                utt = generate_visitor_utterance(None, state, "painting", rng)
                print(f"  - {utt}")
    
    # Test Sim8 response types
    print("\n--- Sim8 Response Types ---")
    for rtype in ["question", "statement", "confusion", "acknowledgment"]:
        if rtype in SIM8_RESPONSES:
            responses = SIM8_RESPONSES[rtype]
            if isinstance(responses, dict):
                total = sum(len(v) for v in responses.values())
            else:
                total = len(responses)
            print(f"\n{rtype.upper()} ({total} templates):")
            for _ in range(3):
                utt = generate_visitor_utterance(rtype, None, "Delft_Masterpiece", rng)
                print(f"  - {utt}")
    
    # Test responsiveness
    print("\n--- Responsiveness Test ---")
    print("After ExplainNewFact:")
    for _ in range(3):
        utt = generate_visitor_utterance(
            None, "HIGHLY_ENGAGED", "painting", rng,
            agent_option="Explain", agent_subaction="ExplainNewFact"
        )
        print(f"  - {utt}")
    
    # Count unique utterances
    print("\n--- Variety Statistics ---")
    for state in VISITOR_RESPONSES:
        responses = VISITOR_RESPONSES[state]
        if isinstance(responses, dict):
            total = sum(len(v) for v in responses.values())
            print(f"{state}: {total} total templates")
        else:
            print(f"{state}: {len(responses)} templates")
    
    for rtype in SIM8_RESPONSES:
        responses = SIM8_RESPONSES[rtype]
        if isinstance(responses, dict):
            total = sum(len(v) for v in responses.values())
            print(f"{rtype}: {total} total templates")
        else:
            print(f"{rtype}: {len(responses)} templates")
