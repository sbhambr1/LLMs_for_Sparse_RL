(define (problem prob)
    (:domain minecraft)
    (:objects
        wood0 wood1 - wood)
    (:init
        (at-starting-location))
    (:goal
        (and (ladder_made))
))