using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Goal : MonoBehaviour
{
    public bool goal_on = false;

    void OnTriggerEnter(Collider col)
    {
        // Goal
        goal_on = true;
    }
}
