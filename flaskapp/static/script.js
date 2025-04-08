document.addEventListener('DOMContentLoaded', () => {
    const submitBtn = document.getElementById('submit-btn');
    submitBtn.addEventListener('click', handleSubmit);
    
    // Also handle Enter key in input field
    const userInput = document.getElementById('user-input');
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleSubmit();
        }
    });
});

async function handleSubmit() {
    const userInput = document.getElementById('user-input').value.trim();
    const outputDiv = document.getElementById('output');
    const vizContainer = document.getElementById('visualization-container');
    const graphDiv = document.getElementById('graph');
    const sqlContainer = document.getElementById('sql-container');
    const dataContainer = document.getElementById('data-container');
    
    // Clear previous results and show loading
    outputDiv.innerHTML = '<p class="loading">Processing your query...</p>';
    vizContainer.style.display = 'none';
    sqlContainer.style.display = 'none';
    dataContainer.style.display = 'none';
    graphDiv.innerHTML = ''; // Clear previous visualization
    
    if (!userInput) {
        outputDiv.innerHTML = '<p class="error">Please enter a question!</p>';
        return;
    }
    
    try {
        console.log('Making API call with query:', userInput);
        
        const response = await fetch('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: userInput })
        });
        
        console.log('Received response status:', response.status);
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        const result = await response.json();
        console.log('Parsed result:', result);
        
        // Update output
        outputDiv.innerHTML = `
            <p><strong>Status:</strong> <span class="${result.status === 'success' ? 'success' : 'error'}">${result.status}</span></p>
            <p><strong>Answer:</strong> ${result.answer || 'No answer generated'}</p>
            ${result.error ? `<p class="error"><strong>Error:</strong> ${result.error}</p>` : ''}
        `;
        
        // Show SQL if available
        if (result.query) {
            document.getElementById('sql-query').textContent = result.query;
            sqlContainer.style.display = 'block';
        }
        
        // Show visualization if available (now handling base64 image)
        if (result.visualization) {
            const img = new Image();
            img.src = `data:image/png;base64,${result.visualization}`;
            img.onload = () => {
                graphDiv.innerHTML = '';
                graphDiv.appendChild(img);
            };
            img.onerror = () => {
                console.error('Failed to load visualization image');
                graphDiv.innerHTML = '<p>Visualization could not be loaded</p>';
            };
            vizContainer.style.display = 'block';
        }
        
        
    } catch (error) {
        console.error('Error:', error);
        outputDiv.innerHTML = `
            <p class="error"><strong>Error:</strong> ${error.message}</p>
            <p>Please check your connection and try again.</p>
        `;
    }
}

function renderDataTable(data) {
    const container = document.getElementById('data-table');
    
    try {
        // Handle different data formats
        let parsedData;
        if (typeof data === 'string') {
            try {
                parsedData = JSON.parse(data);
            } catch {
                // If not JSON, display as-is
                container.innerHTML = `<pre>${data}</pre>`;
                return;
            }
        } else {
            parsedData = data;
        }
        
        // Convert to array if needed
        const dataArray = Array.isArray(parsedData) ? parsedData : [parsedData];
        
        if (dataArray.length === 0) {
            container.innerHTML = '<p>No data available</p>';
            return;
        }
        
        // Create table
        let html = '<table><thead><tr>';
        
        // Add headers
        const headers = Object.keys(dataArray[0]);
        headers.forEach(header => {
            html += `<th>${header}</th>`;
        });
        html += '</tr></thead><tbody>';
        
        // Add rows
        dataArray.forEach(row => {
            html += '<tr>';
            headers.forEach(header => {
                const cellValue = row[header];
                html += `<td>${cellValue !== null && cellValue !== undefined ? cellValue : 'NULL'}</td>`;
            });
            html += '</tr>';
        });
        
        html += '</tbody></table>';
        container.innerHTML = html;
        
    } catch (e) {
        console.error('Error rendering table:', e);
        container.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
    }
}