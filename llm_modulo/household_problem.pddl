(define (problem prob)
    (:domain household)
    (:objects
        key0 key1 - key
        door0 door1 - door)
    (:init
        (at-starting-location))
    (:goal
        (and (at-destination))
))