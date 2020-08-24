using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Ball : MonoBehaviour
{
    public bool goal = false;
    Rigidbody rBody;
    public GameObject ground;
    [HideInInspector] public Bounds areaBounds;

    void Start()
    {
        rBody = GetComponent<Rigidbody>();
    }

    void OnCollisionEnter(Collision col)
    {
        // Goal
        if (col.gameObject.CompareTag("purpleGoal"))
        {
            goal = true;
        }
    }

    public Vector3 randomPosition()
    {
        return ground.transform.position + new Vector3(7.5f, 0f, Random.Range(-6.5f, 6.5f));
    }

    public void resetBall()
    {
        this.transform.position = randomPosition();
        this.transform.rotation = Quaternion.Euler(Vector3.zero);
        rBody.velocity = Vector3.zero;
        rBody.angularVelocity = Vector3.zero;
        goal = false;
    }
}
