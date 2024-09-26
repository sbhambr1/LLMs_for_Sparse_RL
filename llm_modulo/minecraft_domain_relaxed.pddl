(define (domain minecraft_relaxed)
    (:requirements :strips :typing :negative-preconditions)
    (:types wood - object)
    (:predicates 
                (wood-picked ?w - wood)
                (wood-processed ?w - wood)
                (at-starting-location)
                (plank_made)
                (stick_made)
                (ladder_made)
                (processed-to-plank ?w - wood)
                (processed-to-stick ?w - wood))
    (:action get_wood0
            :parameters (?w - wood)
            :precondition (not (wood-picked ?w))
            :effect (and (wood-picked ?w)))
    (:action get_processed_wood
            :parameters (?w - wood)
            :precondition (and (wood-picked ?w)(not (wood-processed ?w)))
            :effect (and (wood-processed ?w)))        
    (:action make_plank
            :parameters (?w - wood)
            :precondition (and (wood-processed ?w) (not (processed-to-plank ?w)) (not (processed-to-stick ?w)))
            :effect (and (processed-to-plank ?w)(plank_made)))
    (:action make_stick
            :parameters (?w - wood)
            :precondition (and (wood-processed ?w) (not (processed-to-stick ?w)) (not (processed-to-plank ?w)))
            :effect (and (processed-to-stick ?w)(stick_made)))
    (:action make_ladder
            :parameters ()
            :precondition (and (stick_made) (plank_made))
            :effect (and (ladder_made)))
)