(define (domain minecraft)
    (:requirements :strips :typing)
    (:types wood - object)
    (:predicates (wood0-picked)
                (wood1-picked)
                (wood0-processed)
                (wood1-processed)
                (at-starting-location)
                (plank_made)
                (stick_made)
                (ladder_made))
    (:action get_wood0
            :parameters ()
            :precondition (and )
            :effect (and (wood0-picked)))
    (:action get_wood1
            :parameters ()
            :precondition (and )
            :effect (and (wood1-picked)))
    (:action get_processed_wood0
            :parameters ()
            :precondition (and (wood0-picked))
            :effect (and (wood0-processed)))        
    (:action get_processed_wood1
            :parameters ()
            :precondition (and (wood1-picked))
            :effect (and (wood1-processed))) 
    (:action make_plank
            :parameters ()
            :precondition (and (or (wood0-processed) (wood1-processed)))
            :effect (and (plank_made)
                         (not (wood0-processed))
                         (not (wood1-processed))))

;     (:action make_plank
;             :parameters ()
;             :precondition (and (wood0-processed)(wood1-processed))
;             :effect (and (plank_made)))
;     (:action make_stick
;             :parameters ()
;             :precondition (and (wood0-processed)(wood1-processed))
;             :effect (and (stick_made)))
    (:action make_ladder
            :parameters ()
            :precondition (and (stick_made) (plank_made))
            :effect (and (ladder_made)))
)