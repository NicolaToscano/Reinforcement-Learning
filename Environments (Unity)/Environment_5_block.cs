using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Block : MonoBehaviour
{
    public bool block_ok = false;
    public bool block_fail = false;

    void OnTriggerStay(Collider col)
    {
        if (this.gameObject.CompareTag("block_orange") && col.gameObject.CompareTag("zone_orange"))
        {
            block_ok = true;
        }

        if (this.gameObject.CompareTag("block_red") && col.gameObject.CompareTag("zone_red"))
        {
            block_ok = true;
        }

        if (this.gameObject.CompareTag("block_orange") && col.gameObject.CompareTag("zone_red"))
        {
            block_fail = true;
        }

        if (this.gameObject.CompareTag("block_red") && col.gameObject.CompareTag("zone_orange"))
        {
            block_fail = true;
        }

    }

    void OnTriggerExit(Collider col)
    {
        if (block_ok)
        {
            block_fail = true;
        }
    }
}
