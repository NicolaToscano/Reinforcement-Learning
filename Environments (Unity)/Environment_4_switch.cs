using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Switch : MonoBehaviour
{
    public bool switch_on = false;

    void OnTriggerStay(Collider col)
    {
        // Switch on
        switch_on = true;
    }

    void OnTriggerExit(Collider col)
    {
        // Switch off
        switch_on = false;
    }
}
