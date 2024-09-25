(define (problem prob)
    (:domain minecraft_relaxed)
    (:objects
        wood0 wood1 - wood)
    (:init
        (at-starting-location))
    (:goal
        (and (ladder_made))
))