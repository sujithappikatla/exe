#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <limits.h>
#include <iostream>
#include <algorithm>

using namespace std;

#define MAX_N 1000007

int segTree[4 * MAX_N];
int lazy[4 * MAX_N];
int arr[MAX_N];

void build(int start, int end, int ind)
{
	lazy[ind] = 0;
	if (start == end)
	{
		segTree[ind] = arr[start];
		return;
	}

	int mid = start + (end - start) / 2;
	build(start, mid, 2 * ind + 1);
	build(mid + 1, end, 2 * ind + 2);
	segTree[ind] = min(segTree[2 * ind + 1], segTree[2 * ind + 2]);
}


void update(int curr_start, int curr_end, int tar_start, int tar_end, int ind, int val)
{
	if (curr_start != curr_end)
	{
		lazy[2 * ind + 1] += lazy[ind];
		lazy[2 * ind + 2] += lazy[ind];
		lazy[ind] = 0;
	}

	if (curr_start > tar_end || curr_end < tar_start)
		return;


	if (curr_start >= tar_start && curr_end <= tar_end)
	{
		lazy[ind] += val;
		return;
	}


	int curr_mid = curr_start + (curr_end - curr_start) / 2;
	update(curr_start, curr_mid, tar_start, tar_end, 2 * ind + 1, val);
	update(curr_mid+1, curr_end, tar_start, tar_end, 2 * ind + 2, val);

	segTree[ind] = min(segTree[2 * ind + 1] + lazy[2*ind+1], segTree[2 * ind + 2] + lazy[2*ind+2]);

}


int query(int curr_start, int curr_end, int tar_start, int tar_end, int ind, int minimum )
{

	if (curr_start != curr_end)
	{
		lazy[2 * ind + 1] += lazy[ind];
		lazy[2 * ind + 2] += lazy[ind];
		lazy[ind] = 0;
	}

	if (curr_start > tar_end || curr_end < tar_start)
		return -1;

	if (segTree[ind] + lazy[ind] > minimum)
		return -1;

	if (curr_start == curr_end)
	{
		if (segTree[ind] + lazy[ind] > minimum)
			return -1;
		return segTree[ind] + lazy[ind];
	}

	int curr_mid = curr_start + (curr_end - curr_start) / 2;
	int ret = query(curr_start, curr_mid, tar_start, tar_end, 2 * ind + 1, minimum);
	if (ret == -1)
		ret = query(curr_mid + 1, curr_end, tar_start, tar_end, 2 * ind + 2, minimum);

	return ret;
}

int point_query(int curr_start, int curr_end, int tar_start, int tar_end, int ind)
{
	if (curr_start != curr_end)
	{
		lazy[2 * ind + 1] += lazy[ind];
		lazy[2 * ind + 2] += lazy[ind];
		lazy[ind] = 0;
	}

	if (curr_start > tar_end || curr_end < tar_start)
		return INT_MAX;


	if (curr_start >= tar_start && curr_end <= tar_end)
	{
		return segTree[ind]+lazy[ind];
	}
	int curr_mid = curr_start + (curr_end - curr_start) / 2;
	int ret = point_query(curr_start, curr_mid, tar_start, tar_end, 2 * ind + 1);
	if(ret == INT_MAX)
		ret = point_query(curr_mid + 1, curr_end, tar_start, tar_end, 2 * ind + 2);
	return ret;
}



void solve()
{
	int n, q;
	cin >> n >> q;
	for (int i = 0; i < n; i++)
		cin >> arr[i];

	build(0, n - 1, 0);

	while (q--)
	{
		int type;
		cin >> type;
		if (type == 1)
		{
			int a, b, v;
			cin >> a >> b >> v;
			update(0, n - 1, a, b, 0, v);
		}
		else
		{
			int c, b;
			cin >> c >> b;
			
			if (c > b) swap(c, b);
			int vc = point_query(0, n - 1, c, c, 0);
			int vb = point_query(0, n - 1, b, b, 0);

			if (vc > vb)
			{
				cout << vb << endl;
				continue;
			}
			int ret = query(0, n - 1, b, n - 1, 0, min(vb, vc)-1);
			cout << ret << endl;
		}
	}
}


int main()
{
	//freopen("input.txt", "r", stdin);

	int ignore;
	cin >> ignore;

	int tc;
	cin >> tc;
	while (tc--)
	{
		solve();
	}
	return 0;
}
